import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
from Heat_1D_PINO import FNO1d, SpectralConv1d

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# --- 1. DeepONet Model Definition ---

class DeepONet1d(nn.Module):
    def __init__(self, num_sensors, p_dim=128):
        super(DeepONet1d, self).__init__()
        self.num_sensors = num_sensors
        self.p_dim = p_dim # Output dimension of branch/trunk

        # Branch Net: Takes the initial condition u0(x) at 'num_sensors' points
        self.branch = nn.Sequential(
            nn.Linear(num_sensors, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, self.p_dim),
        )

        # Trunk Net: Takes the coordinate 'x' (1D)
        self.trunk = nn.Sequential(
            nn.Linear(1, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, self.p_dim),
        )

        # Bias (optional but often helps)
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_in):
        # x_in format from your data generator is [Batch, NX, 2] where channel 0 is u0, channel 1 is coordinates.
        # DeepONet Branch needs just u0: [Batch, NX]
        u0 = x_in[..., 0]
        # Trunk needs coordinates. We can extract them from the first sample.
        # In this fixed grid case, we can just use one set of coordinates for all batches.
        coords = x_in[0, :, 1].unsqueeze(-1) # Shape: [NX, 1]

        # Forward pass
        B_out = self.branch(u0)      # Shape: [Batch, p_dim]
        T_out = self.trunk(coords)   # Shape: [NX, p_dim]

        # Efficient DeepONet combination for fixed grid: (B x T^T)
        # [Batch, p] x [p, NX] -> [Batch, NX]
        outputs = torch.matmul(B_out, T_out.T) + self.b

        return outputs.unsqueeze(-1) # match FNO output shape [Batch, NX, 1]

# --- 2. Shared Data Generation (Re-using your functions for consistency) ---
# (Assuming generate_heat_equation_data is already defined in your previous block)
# Re-defining here briefly just to ensure standalone runnability if needed,
# but in a notebook, you can skip this block if already run.

def heat_equation_fd_solver(u0, nx, nt, alpha, dt):
    dx = 2.0 / (nx - 1)
    u = u0.clone()
    for _ in range(nt):
        un = u.clone()
        u_xx = (un[2:] - 2 * un[1:-1] + un[:-2]) / (dx ** 2)
        u[1:-1] = un[1:-1] + alpha * dt * u_xx
        u[0] = 0.0; u[-1] = 0.0
    return u

def generate_data_shared(num_samples, nx, nt, alpha=0.01):
    x = torch.linspace(-1, 1, nx)
    dt = 1.0 / nt
    u0_data = torch.zeros(num_samples, nx)
    for i in range(num_samples):
        num_waves = torch.randint(1, 4, (1,)).item()
        u0 = torch.zeros_like(x)
        for _ in range(num_waves):
            k = torch.randint(1, 6, (1,)).item()
            u0 += (0.5 + 0.5 * torch.rand(1).item()) * torch.sin(k * np.pi * x + 2 * np.pi * torch.rand(1).item())
        u0_data[i] = (u0 - u0[0]) * (1.0 - torch.abs(x))
    u_final = torch.zeros_like(u0_data)
    for i in range(num_samples):
        u_final[i] = heat_equation_fd_solver(u0_data[i], nx, nt, alpha, dt)
    return torch.stack([u0_data, x.repeat(num_samples, 1)], dim=-1).to(device), u_final.unsqueeze(-1).to(device)

# --- 3. Generic PINO Training Loop (Works for both models) ---

def train_model_generic(model, train_loader, epochs=1000, pde_weight=0.1, model_name="Model"):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
    history = {'total': [], 'data': [], 'pde': []}
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        ep_loss, ep_data, ep_pde = 0., 0., 0.
        for ic, sol in train_loader:
            optimizer.zero_grad()
            pred = model(ic)
            loss_d = F.mse_loss(pred, sol)

            # --- Identical coarse PDE loss from your example ---
            u_pred = pred.squeeze(-1)
            u_init = ic[..., 0]
            nx = ic.shape[1]
            dx = 2.0 / (nx - 1)
            # Coarse time derivative approx over full interval t=[0,1]
            u_t = (u_pred - u_init) / 1.0
            u_xx = (u_pred[:, 2:] - 2 * u_pred[:, 1:-1] + u_pred[:, :-2]) / (dx ** 2)
            pde_res = u_t[:, 1:-1] - 0.01 * u_xx
            loss_p = F.mse_loss(pde_res, torch.zeros_like(pde_res))
            # ---------------------------------------------------

            loss = loss_d + pde_weight * loss_p
            loss.backward()
            optimizer.step()
            ep_loss += loss.item(); ep_data += loss_d.item(); ep_pde += loss_p.item()

        scheduler.step()
        history['total'].append(ep_loss / len(train_loader))
        history['data'].append(ep_data / len(train_loader))
        history['pde'].append(ep_pde / len(train_loader))
        if (epoch + 1) % 100 == 0:
            print(f"[{model_name}] Ep {epoch+1}/{epochs} | Total: {history['total'][-1]:.4e} | Data: {history['data'][-1]:.4e} | PDE: {history['pde'][-1]:.4e}")

    print(f"[{model_name}] Training finished in {time.time()-start_time:.2f}s")
    return model, history

# --- 4. Execution & Comparison ---

# Setup Data
NX = 128
N_TRAIN = 200
N_TEST = 50
BATCH_SIZE = 32
print(f"\n--- Generating {NX}-resolution data ---")
train_ic, train_sol = generate_data_shared(N_TRAIN, NX, 500)
test_ic, test_sol = generate_data_shared(N_TEST, NX, 500)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_ic, train_sol), batch_size=BATCH_SIZE, shuffle=True)

# A. Train DeepONet
print("\n--- Training DeepONet ---")
don_model = DeepONet1d(num_sensors=NX, p_dim=64).to(device)
don_model, don_hist = train_model_generic(don_model, train_loader, epochs=500, pde_weight=0.01, model_name="DeepONet")

#Visualization DeepONet

# --- DeepONet Specific Visualizations ---
import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_deeponet_results(model, test_ic, test_sol, history):
    model.eval()
    with torch.no_grad():
        # Select one test case to visualize (e.g., index 0)
        idx = 0
        initial_cond = test_ic[idx:idx + 1]
        true_solution = test_sol[idx]
        pred_solution = model(initial_cond).squeeze()

    # Detach and move to CPU for plotting
    # Assuming second channel of input is coordinates, if available, else generate them
    if initial_cond.shape[-1] == 2:
         x = initial_cond[0, :, 1].cpu()
         ic_plot = initial_cond[0, :, 0].cpu()
    else:
         # Fallback if generic data loader changed shape
         x = torch.linspace(-1, 1, initial_cond.shape[1])
         ic_plot = initial_cond[0].cpu()

    true_sol_plot = true_solution.squeeze().cpu()
    pred_sol_plot = pred_solution.cpu()

    # --- FIGURE 1: Qualitative Results (3 Subplots) ---
    plt.figure(figsize=(18, 5))

    # Plot 1: Initial Condition
    plt.subplot(1, 3, 1)
    plt.plot(x, ic_plot, label="Initial Condition u(x, 0)", color='black')
    plt.title("DeepONet: Input (Initial Condition)")
    plt.xlabel("x"); plt.ylabel("u(x,0)")
    plt.legend(); plt.grid(True, linestyle='--')

    # Plot 2: True vs. Predicted Solution
    plt.subplot(1, 3, 2)
    plt.plot(x, true_sol_plot, label="Ground Truth u(x, 1)", color='blue')
    plt.plot(x, pred_sol_plot, label="DeepONet Prediction u(x, 1)", color='red', linestyle='--')
    plt.title("DeepONet: Solution at t=1")
    plt.xlabel("x"); plt.ylabel("u(x,1)")
    plt.legend(); plt.grid(True, linestyle='--')

    # Plot 3: Error
    plt.subplot(1, 3, 3)
    error = torch.abs(true_sol_plot - pred_sol_plot)
    plt.plot(x, error, label="Absolute Error", color='green')
    plt.title("DeepONet: Prediction Error")
    plt.xlabel("x"); plt.ylabel("|True - Predicted|")
    plt.legend(); plt.grid(True, linestyle='--')

    plt.tight_layout()
    plt.show()

    # --- FIGURE 2: Training History ---
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    # Handle generic history dict from the comparison script
    plt.plot(history['total'], label='Total Loss', color='black', linewidth=2)
    plt.plot(history['data'], label='Data Loss', color='blue', alpha=0.7)
    plt.plot(history['pde'], label='PDE Loss', color='red', alpha=0.7)
    plt.title('DeepONet Training Loss History')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.yscale('log')
    plt.legend(); plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(history['total'], label='Total Loss', color='black', linewidth=2)
    plt.title('DeepONet Total Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.yscale('log')
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.show()

# Run the visualization using the DeepONet model and its history
print("Visualizing DeepONet results...")
visualize_deeponet_results(don_model, test_ic, test_sol, don_hist)

# B. Train FNO (re-instantiating your model for fair fresh comparison)
# Need to re-define FNO classes if not in same active session, assuming they are present from your prompt.
# (I will assume FNO1d and SpectralConv1d are available from your provided code)
print("\n--- Training FNO ---")
# Note: Using generic trainer to ensure EXACT same loss calculation
fno_model = FNO1d(modes=16, width=64).to(device)
fno_model, fno_hist = train_model_generic(fno_model, train_loader, epochs=500, pde_weight=0.01, model_name="FNO")

# --- 5. Comparative Evaluation ---

def get_errors(model, ic, sol):
    model.eval()
    with torch.no_grad():
        pred = model(ic)
        mse = F.mse_loss(pred, sol, reduction='none').mean(dim=(1,2)) # MSE per sample
        rel_l2 = torch.norm(pred - sol, p=2, dim=(1,2)) / torch.norm(sol, p=2, dim=(1,2))
    return mse.cpu().numpy(), rel_l2.cpu().numpy(), pred

don_mse, don_l2, don_pred = get_errors(don_model, test_ic, test_sol)
fno_mse, fno_l2, fno_pred = get_errors(fno_model, test_ic, test_sol)

print("\n--- Final Test Results (Average over 50 samples) ---")
print(f"DeepONet | MSE: {np.mean(don_mse):.4e} | Relative L2 Error: {np.mean(don_l2):.4%}")
print(f"FNO      | MSE: {np.mean(fno_mse):.4e} | Relative L2 Error: {np.mean(fno_l2):.4%}")

# --- 6. Comparative Visualization ---
idx = 0 # Pick first test sample
x_grid = torch.linspace(-1, 1, NX)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Loss History Comparison")
plt.plot(don_hist['total'], label='DeepONet Total', color='blue', alpha=0.7)
plt.plot(fno_hist['total'], label='FNO Total', color='red', alpha=0.7)
plt.yscale('log')
plt.xlabel('Epochs'); plt.ylabel('Loss')
plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.title(f"Prediction (Test Sample {idx})")
plt.plot(x_grid, test_sol[idx].cpu().squeeze(), 'k-', linewidth=2, label='Ground Truth')
plt.plot(x_grid, don_pred[idx].cpu().squeeze(), 'b--', label='DeepONet')
plt.plot(x_grid, fno_pred[idx].cpu().squeeze(), 'r:', linewidth=2, label='FNO')
plt.xlabel('x'); plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.title("Absolute Error Comparison")
plt.plot(x_grid, torch.abs(test_sol[idx] - don_pred[idx]).cpu().squeeze(), 'b-', label='DeepONet Error')
plt.plot(x_grid, torch.abs(test_sol[idx] - fno_pred[idx]).cpu().squeeze(), 'r--', label='FNO Error')
plt.xlabel('x'); plt.yscale('log'); plt.legend(); plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()