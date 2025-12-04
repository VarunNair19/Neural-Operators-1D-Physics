'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time

# Check for GPU and set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


# --- PINO Model Definition: Fourier Neural Operator (FNO) ---
# This model is modified to handle the 1D Wave Equation system

class SpectralConv1d(nn.Module):
    """1D Fourier layer. It does FFT, linear transform, and Inverse FFT."""

    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = torch.einsum("bix,iox->box", x_ft[:, :, :self.modes1], self.weights1)

        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(nn.Module):
    """
    Modified FNO for the 1D Wave Equation (u_tt = c^2 * u_xx)
    We solve it as a system:
    1. u_t = v
    2. v_t = c^2 * u_xx
    """

    def __init__(self, modes, width):
        super(FNO1d, self).__init__()
        self.modes1 = modes
        self.width = width

        # *** MODIFIED ***
        # Input: [u(x,0), v(x,0), x] (3 channels)
        self.fc0 = nn.Linear(3, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)

        # *** MODIFIED ***
        # Output: [u(x,T), v(x,T)] (2 channels)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


# --- Data Generation for 1D Wave Equation ---

def wave_equation_fd_solver(u0, v0, nx, nt, c, dt):
    """
    Solves the 1D wave equation system:
    u_t = v
    v_t = c^2 * u_xx
    Uses a simple Forward Euler for time, Central Difference for space.
    """
    dx = 2.0 / (nx - 1)
    u = u0.clone()
    v = v0.clone()

    for _ in range(nt):
        un = u.clone()
        vn = v.clone()

        # Central difference for u_xx
        u_xx = (un[2:] - 2 * un[1:-1] + un[:-2]) / (dx ** 2)

        # Update v using forward Euler
        v[1:-1] = vn[1:-1] + c ** 2 * dt * u_xx

        # Update u using forward Euler
        u[1:-1] = un[1:-1] + dt * vn[1:-1]  # Use vn, not v_new

        # Boundary conditions (Dirichlet)
        u[0] = 0.0
        u[-1] = 0.0
        v[0] = 0.0
        v[-1] = 0.0

    return u, v


def generate_wave_equation_data(num_samples, nx, nt, c=1.0, total_time=1.0):
    """Generates data for 1D Wave equation"""
    x = torch.linspace(-1, 1, nx)
    dt = total_time / nt

    # Generate diverse initial positions u0
    u0_data = torch.zeros(num_samples, nx)
    for i in range(num_samples):
        num_waves = torch.randint(1, 4, (1,)).item()
        u0 = torch.zeros_like(x)
        for j in range(num_waves):
            k = torch.randint(1, 6, (1,)).item()
            amplitude = 0.5 + 0.5 * torch.rand(1).item()
            phase = 2 * np.pi * torch.rand(1).item()
            u0 += amplitude * torch.sin(k * np.pi * x + phase)

        # Enforce zero boundaries
        u0[0] = 0.0
        u0[-1] = 0.0
        u0_data[i] = u0

    # Generate initial velocities v0 (released from rest)
    v0_data = torch.zeros(num_samples, nx)

    # Solve wave equation for each initial condition
    u_final = torch.zeros_like(u0_data)
    v_final = torch.zeros_like(v0_data)

    for i in range(num_samples):
        u_f, v_f = wave_equation_fd_solver(u0_data[i], v0_data[i], nx, nt, c, dt)
        u_final[i] = u_f
        v_final[i] = v_f

    # Add spatial coordinate as a channel
    # Input shape: [u0, v0, x]
    initial_conditions = torch.stack([u0_data, v0_data, x.repeat(num_samples, 1)], dim=-1)

    # Target shape: [u_final, v_final]
    final_solutions = torch.stack([u_final, v_final], dim=-1)

    print(f"  Data generation complete.")
    print(f"  Solution u range: [{torch.min(u_final):.3f}, {torch.max(u_final):.3f}]")
    print(f"  Solution v range: [{torch.min(v_final):.3f}, {torch.max(v_final):.3f}]")

    return initial_conditions.to(device), final_solutions.to(device)


# --- Training the PINO model ---

def train_pino(model, train_loader, epochs=500, pde_weight=0.1):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

    loss_history = []
    data_loss_history = []
    pde_loss_history = []

    c = 1.0  # Wave speed from PDF (c = sqrt(E/rho))
    c_squared = c ** 2
    total_time = 1.0  # Total time T for the operator

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        data_loss_total = 0.0
        pde_loss_total = 0.0
        batch_count = 0

        for initial_conditions, final_solutions in train_loader:
            optimizer.zero_grad()

            # --- 1. Data Loss (Supervised) ---
            # This now compares both u and v at the final time
            pred_final = model(initial_conditions)
            loss_data = F.mse_loss(pred_final, final_solutions)

            # --- 2. Physics Loss (PINO) ---
            # We check the residuals of the *two* 1st-order PDEs

            # Get initial and predicted states
            u_pred = pred_final[..., 0].squeeze(-1)
            v_pred = pred_final[..., 1].squeeze(-1)

            u_initial = initial_conditions[..., 0]
            v_initial = initial_conditions[..., 1]

            # --- PDE 1: u_t - v = 0 ---
            # Approx u_t at T as (u_T - u_0) / T
            u_t_approx = (u_pred - u_initial) / total_time
            # Residual 1: u_t(T) - v(T)
            pde_residual_1 = u_t_approx - v_pred

            # --- PDE 2: v_t - c^2 * u_xx = 0 ---
            # Approx v_t at T as (v_T - v_0) / T
            v_t_approx = (v_pred - v_initial) / total_time

            # Approx u_xx at T using finite differences on u_pred
            nx = initial_conditions.shape[1]
            dx = 2.0 / (nx - 1)
            u_xx = (u_pred[:, 2:] - 2 * u_pred[:, 1:-1] + u_pred[:, :-2]) / (dx ** 2)

            # Trim v_t to match u_xx's spatial shape
            v_t_trimmed = v_t_approx[:, 1:-1]

            # Residual 2: v_t(T) - c^2 * u_xx(T)
            pde_residual_2 = v_t_trimmed - c_squared * u_xx

            # Combine PDE losses
            loss_pde = F.mse_loss(pde_residual_1, torch.zeros_like(pde_residual_1)) + \
                       F.mse_loss(pde_residual_2, torch.zeros_like(pde_residual_2))

            # --- Total loss ---
            total_loss = loss_data + pde_weight * loss_pde

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            data_loss_total += loss_data.item()
            pde_loss_total += loss_pde.item()
            batch_count += 1

        scheduler.step()

        # Calculate average losses
        epoch_loss /= batch_count
        data_loss_total /= batch_count
        pde_loss_total /= batch_count

        loss_history.append(epoch_loss)
        data_loss_history.append(data_loss_total)
        pde_loss_history.append(pde_loss_total)

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}/{epochs}, Total Loss: {epoch_loss:.6f}, "
                  f"Data Loss: {data_loss_total:.6f}, PDE Loss: {pde_loss_total:.6f}")

    return model, loss_history, data_loss_history, pde_loss_history


# --- Main Execution ---

# Parameters
N_TRAIN = 200
N_TEST = 20
# NX = 128  <-- THIS CAUSES INSTABILITY
NX = 32  # <-- STABLE. r = (c^2 * dt / dx^2) = (1*0.002 / (2/31)^2) = 0.48 <= 0.5
NT = 500
BATCH_SIZE = 16
C_WAVE = 1.0  # c = sqrt(E/rho)
TOTAL_TIME = 1.0  # Operator maps from t=0 to t=1

# --- FDM Stability Check ---
dx = 2.0 / (NX - 1)
dt = TOTAL_TIME / NT
r_stability = (C_WAVE ** 2 * dt) / (dx ** 2)
print("--- FDM Stability Check ---")
print(f"  NX = {NX}, NT = {NT}")
print(f"  dx = {dx:.5f}, dt = {dt:.5f}")
print(f"  Stability Parameter r = (c^2 * dt) / dx^2 = {r_stability:.5f}")
if r_stability > 0.5:
    print("  WARNING: FDM SOLVER IS UNSTABLE. Loss will explode.")
    print(f"  To fix, set NT >= {int(TOTAL_TIME * C_WAVE ** 2 / (0.5 * dx ** 2)) + 1}")
else:
    print("  FDM Solver is STABLE (r <= 0.5)")
print("---------------------------")

print("Generating training data for 1D Wave Equation...")
train_ic, train_sol = generate_wave_equation_data(N_TRAIN, NX, NT, c=C_WAVE, total_time=TOTAL_TIME)
print(f"Training IC shape: {train_ic.shape}")
print(f"Training Sol shape: {train_sol.shape}")

print("Generating test data...")
test_ic, test_sol = generate_wave_equation_data(N_TEST, NX, NT, c=C_WAVE, total_time=TOTAL_TIME)

train_dataset = torch.utils.data.TensorDataset(train_ic, train_sol)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Create and Train Model
print("Creating model...")
model = FNO1d(modes=16, width=64).to(device)
print("Starting training...")
trained_model, loss_history, data_loss_history, pde_loss_history = train_pino(
    model, train_loader, epochs=1000, pde_weight=0.01
)


# --- Visualization ---

def visualize_results(model, test_ic, test_sol):
    model.eval()
    with torch.no_grad():
        # Select one test case to visualize
        idx = 0
        initial_cond = test_ic[idx:idx + 1]
        true_solution = test_sol[idx]

        pred_solution_stacked = model(initial_cond).squeeze()

    # Detach and move to CPU for plotting
    x = torch.linspace(-1, 1, initial_cond.shape[1])

    # Initial conditions
    u0_plot = initial_cond.squeeze()[..., 0].cpu()
    v0_plot = initial_cond.squeeze()[..., 1].cpu()

    # True solutions at T=1
    uT_true_plot = true_solution.squeeze()[..., 0].cpu()
    vT_true_plot = true_solution.squeeze()[..., 1].cpu()

    # Predicted solutions at T=1
    uT_pred_plot = pred_solution_stacked.squeeze()[..., 0].cpu()
    vT_pred_plot = pred_solution_stacked.squeeze()[..., 1].cpu()

    plt.figure(figsize=(12, 10))

    # Plot 1: Initial Conditions
    plt.subplot(2, 2, 1)
    plt.plot(x, u0_plot, label="u(x, 0) - Initial Position", color='black')
    plt.plot(x, v0_plot, label="v(x, 0) - Initial Velocity", color='gray', linestyle='--')
    plt.title("Input to the Operator (t=0)")
    plt.xlabel("x")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, linestyle='--')

    # Plot 2: Final Position u(x, T)
    plt.subplot(2, 2, 2)
    plt.plot(x, uT_true_plot, label="Ground Truth u(x, T)", color='blue')
    plt.plot(x, uT_pred_plot, label="PINO Prediction u(x, T)", color='red', linestyle='--')
    plt.title("Final Position at t=T")
    plt.xlabel("x")
    plt.ylabel("u(x,T)")
    plt.legend()
    plt.grid(True, linestyle='--')

    # Plot 3: Final Velocity v(x, T)
    plt.subplot(2, 2, 3)
    plt.plot(x, vT_true_plot, label="Ground Truth v(x, T)", color='blue')
    plt.plot(x, vT_pred_plot, label="PINO Prediction v(x, T)", color='red', linestyle='--')
    plt.title("Final Velocity at t=T")
    plt.xlabel("x")
    plt.ylabel("v(x,T)")
    plt.legend()
    plt.grid(True, linestyle='--')

    # Plot 4: Position Error
    plt.subplot(2, 2, 4)
    error_u = torch.abs(uT_true_plot - uT_pred_plot)
    error_v = torch.abs(vT_true_plot - vT_pred_plot)
    plt.plot(x, error_u, label="Position Error |u_true - u_pred|", color='green')
    plt.plot(x, error_v, label="Velocity Error |v_true - v_pred|", color='orange', linestyle='--')
    plt.title("Prediction Error at t=T")
    plt.xlabel("x")
    plt.ylabel("Absolute Error")
    plt.legend()
    plt.grid(True, linestyle='--')

    plt.tight_layout()
    plt.show()


print("\nVisualizing results on a new, unseen initial condition...")
visualize_results(trained_model, test_ic, test_sol)

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='Total Loss', color='black', linewidth=2)
plt.plot(data_loss_history, label='Data Loss', color='blue', alpha=0.7)
plt.plot(pde_loss_history, label='PDE Loss', color='red', alpha=0.7)
plt.title('Training Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time

# Check for GPU and set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


# --- PINO Model Definition: Fourier Neural Operator (FNO) ---
# This model is modified to handle the 1D Wave Equation system

class SpectralConv1d(nn.Module):
    """1D Fourier layer. It does FFT, linear transform, and Inverse FFT."""

    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = torch.einsum("bix,iox->box", x_ft[:, :, :self.modes1], self.weights1)

        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(nn.Module):
    """
    Modified FNO for the 1D Wave Equation (u_tt = c^2 * u_xx)
    We solve it as a system:
    1. u_t = v
    2. v_t = c^2 * u_xx
    """

    def __init__(self, modes, width):
        super(FNO1d, self).__init__()
        self.modes1 = modes
        self.width = width

        # *** MODIFIED ***
        # Input: [u(x,0), v(x,0), x] (3 channels)
        self.fc0 = nn.Linear(3, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)

        # Output: [u(x,T), v(x,T)] (2 channels)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


# --- Data Generation for 1D Wave Equation ---

def wave_equation_fd_solver(u0, v0, nx, nt, c, dt):
    """
    Solves the 1D wave equation system:
    u_t = v
    v_t = c^2 * u_xx
    Uses a simple Forward Euler for time, Central Difference for space.
    """
    dx = 2.0 / (nx - 1)
    u = u0.clone()
    v = v0.clone()

    for _ in range(nt):
        un = u.clone()
        vn = v.clone()

        # Central difference for u_xx
        u_xx = (un[2:] - 2 * un[1:-1] + un[:-2]) / (dx ** 2)

        # Update v using forward Euler
        v[1:-1] = vn[1:-1] + c ** 2 * dt * u_xx

        # Update u using forward Euler
        u[1:-1] = un[1:-1] + dt * vn[1:-1]  # Use vn, not v_new

        # Boundary conditions (Dirichlet)
        u[0] = 0.0
        u[-1] = 0.0
        v[0] = 0.0
        v[-1] = 0.0

    return u, v


def generate_wave_equation_data(num_samples, nx, nt, c=1.0, total_time=1.0):
    """Generates data for 1D Wave equation"""
    x = torch.linspace(-1, 1, nx)
    dt = total_time / nt

    # Generate diverse initial positions u0
    u0_data = torch.zeros(num_samples, nx)
    for i in range(num_samples):
        num_waves = torch.randint(1, 4, (1,)).item()
        u0 = torch.zeros_like(x)
        for j in range(num_waves):
            k = torch.randint(1, 6, (1,)).item()
            amplitude = 0.5 + 0.5 * torch.rand(1).item()
            phase = 2 * np.pi * torch.rand(1).item()
            u0 += amplitude * torch.sin(k * np.pi * x + phase)

        # Enforce zero boundaries
        u0[0] = 0.0
        u0[-1] = 0.0
        u0_data[i] = u0

    # Generate initial velocities v0 (released from rest)
    v0_data = torch.zeros(num_samples, nx)

    # Solve wave equation for each initial condition
    u_final = torch.zeros_like(u0_data)
    v_final = torch.zeros_like(v0_data)

    for i in range(num_samples):
        u_f, v_f = wave_equation_fd_solver(u0_data[i], v0_data[i], nx, nt, c, dt)
        u_final[i] = u_f
        v_final[i] = v_f

    # Add spatial coordinate as a channel
    # Input shape: [u0, v0, x]
    initial_conditions = torch.stack([u0_data, v0_data, x.repeat(num_samples, 1)], dim=-1)

    # Target shape: [u_final, v_final]
    final_solutions = torch.stack([u_final, v_final], dim=-1)

    print(f"  Data generation complete.")
    print(f"  Solution u range: [{torch.min(u_final):.3f}, {torch.max(u_final):.3f}]")
    print(f"  Solution v range: [{torch.min(v_final):.3f}, {torch.max(v_final):.3f}]")

    return initial_conditions.to(device), final_solutions.to(device)


# --- Training the PINO model ---

def train_pino(model, train_loader, epochs=500, pde_weight=0.1):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

    loss_history = []
    data_loss_history = []
    pde_loss_history = []

    c = 1.0  # Wave speed from PDF (c = sqrt(E/rho))
    c_squared = c ** 2
    total_time = 1.0  # Total time T for the operator

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        data_loss_total = 0.0
        pde_loss_total = 0.0
        batch_count = 0

        for initial_conditions, final_solutions in train_loader:
            optimizer.zero_grad()

            # --- 1. Data Loss (Supervised) ---
            # This now compares both u and v at the final time
            pred_final = model(initial_conditions)

            # We split the data loss for better scaling, but MSELoss handles it
            loss_data = F.mse_loss(pred_final, final_solutions)

            # --- 2. Physics Loss (PINO) ---
            # We check the residuals of the *two* 1st-order PDEs

            # Get initial and predicted states
            u_pred = pred_final[..., 0].squeeze(-1)
            v_pred = pred_final[..., 1].squeeze(-1)

            u_initial = initial_conditions[..., 0]
            v_initial = initial_conditions[..., 1]

            # --- PDE 1: u_t - v = 0 ---
            # Approx u_t at T as (u_T - u_0) / T
            u_t_approx = (u_pred - u_initial) / total_time
            # Residual 1: u_t(T) - v(T)
            pde_residual_1 = u_t_approx - v_pred

            # --- PDE 2: v_t - c^2 * u_xx = 0 ---
            # Approx v_t at T as (v_T - v_0) / T
            v_t_approx = (v_pred - v_initial) / total_time

            # Approx u_xx at T using finite differences on u_pred
            nx = initial_conditions.shape[1]
            dx = 2.0 / (nx - 1)
            u_xx = (u_pred[:, 2:] - 2 * u_pred[:, 1:-1] + u_pred[:, :-2]) / (dx ** 2)

            # Trim v_t to match u_xx's spatial shape
            v_t_trimmed = v_t_approx[:, 1:-1]
            c_squared_u_xx = c_squared * u_xx

            # Residual 2: v_t(T) - c^2 * u_xx(T)
            pde_residual_2 = v_t_trimmed - c_squared_u_xx

            # --- Combine PDE losses with Normalization ---
            # This is the key fix. We scale each loss by the
            # variance (MSE) of its own terms, making them comparable.

            # L1 loss: ||u_t_approx - v_pred||^2 / ||v_pred||^2
            # Add epsilon to denominator to prevent division by zero
            loss_pde_1 = F.mse_loss(pde_residual_1, torch.zeros_like(pde_residual_1)) / \
                         (F.mse_loss(v_pred, torch.zeros_like(v_pred)) + 1e-8)

            # L2 loss: ||v_t_trimmed - c^2*u_xx||^2 / ||c^2*u_xx||^2
            loss_pde_2 = F.mse_loss(pde_residual_2, torch.zeros_like(pde_residual_2)) / \
                         (F.mse_loss(c_squared_u_xx, torch.zeros_like(c_squared_u_xx)) + 1e-8)

            loss_pde = loss_pde_1 + loss_pde_2

            # --- Total loss ---
            total_loss = loss_data + pde_weight * loss_pde

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            data_loss_total += loss_data.item()
            pde_loss_total += loss_pde.item()
            batch_count += 1

        scheduler.step()

        # Calculate average losses
        epoch_loss /= batch_count
        data_loss_total /= batch_count
        pde_loss_total /= batch_count

        loss_history.append(epoch_loss)
        data_loss_history.append(data_loss_total)
        pde_loss_history.append(pde_loss_total)

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}/{epochs}, Total Loss: {epoch_loss:.6f}, "
                  f"Data Loss: {data_loss_total:.6f}, PDE Loss: {pde_loss_total:.6f}")

    return model, loss_history, data_loss_history, pde_loss_history


# --- Main Execution ---

# Parameters
N_TRAIN = 200
N_TEST = 20
# NX = 128  <-- THIS CAUSES INSTABILITY
NX = 32  # <-- STABLE. r = (c^2 * dt / dx^2) = (1*0.002 / (2/31)^2) = 0.48 <= 0.5
NT = 500
BATCH_SIZE = 16
C_WAVE = 1.0  # c = sqrt(E/rho)
TOTAL_TIME = 1.0  # Operator maps from t=0 to t=1

# --- FDM Stability Check ---
dx = 2.0 / (NX - 1)
dt = TOTAL_TIME / NT
r_stability = (C_WAVE ** 2 * dt) / (dx ** 2)
print("--- FDM Stability Check ---")
print(f"  NX = {NX}, NT = {NT}")
print(f"  dx = {dx:.5f}, dt = {dt:.5f}")
print(f"  Stability Parameter r = (c^2 * dt) / dx^2 = {r_stability:.5f}")
if r_stability > 0.5:
    print("  WARNING: FDM SOLVER IS UNSTABLE. Loss will explode.")
    print(f"  To fix, set NT >= {int(TOTAL_TIME * C_WAVE ** 2 / (0.5 * dx ** 2)) + 1}")
else:
    print("  FDM Solver is STABLE (r <= 0.5)")
print("---------------------------")

print("Generating training data for 1D Wave Equation...")
train_ic, train_sol = generate_wave_equation_data(N_TRAIN, NX, NT, c=C_WAVE, total_time=TOTAL_TIME)
print(f"Training IC shape: {train_ic.shape}")
print(f"Training Sol shape: {train_sol.shape}")

print("Generating test data...")
test_ic, test_sol = generate_wave_equation_data(N_TEST, NX, NT, c=C_WAVE, total_time=TOTAL_TIME)

train_dataset = torch.utils.data.TensorDataset(train_ic, train_sol)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Create and Train Model
print("Creating model...")
model = FNO1d(modes=16, width=64).to(device)
print("Starting training...")
trained_model, loss_history, data_loss_history, pde_loss_history = train_pino(
    model, train_loader, epochs=1000, pde_weight=0.1  # <-- We can use a reasonable weight again
)


# --- Visualization ---

def visualize_results(model, test_ic, test_sol):
    model.eval()
    with torch.no_grad():
        # Select one test case to visualize
        idx = 0
        initial_cond = test_ic[idx:idx + 1]
        true_solution = test_sol[idx]

        pred_solution_stacked = model(initial_cond).squeeze()

    # Detach and move to CPU for plotting
    x = torch.linspace(-1, 1, initial_cond.shape[1])

    # Initial conditions
    u0_plot = initial_cond.squeeze()[..., 0].cpu()
    v0_plot = initial_cond.squeeze()[..., 1].cpu()

    # True solutions at T=1
    uT_true_plot = true_solution.squeeze()[..., 0].cpu()
    vT_true_plot = true_solution.squeeze()[..., 1].cpu()

    # Predicted solutions at T=1
    uT_pred_plot = pred_solution_stacked.squeeze()[..., 0].cpu()
    vT_pred_plot = pred_solution_stacked.squeeze()[..., 1].cpu()

    plt.figure(figsize=(12, 10))

    # Plot 1: Initial Conditions
    plt.subplot(2, 2, 1)
    plt.plot(x, u0_plot, label="u(x, 0) - Initial Position", color='black')
    plt.plot(x, v0_plot, label="v(x, 0) - Initial Velocity", color='gray', linestyle='--')
    plt.title("Input to the Operator (t=0)")
    plt.xlabel("x")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, linestyle='--')

    # Plot 2: Final Position u(x, T)
    plt.subplot(2, 2, 2)
    plt.plot(x, uT_true_plot, label="Ground Truth u(x, T)", color='blue')
    plt.plot(x, uT_pred_plot, label="PINO Prediction u(x, T)", color='red', linestyle='--')
    plt.title("Final Position at t=T")
    plt.xlabel("x")
    plt.ylabel("u(x,T)")
    plt.legend()
    plt.grid(True, linestyle='--')

    # Plot 3: Final Velocity v(x, T)
    plt.subplot(2, 2, 3)
    plt.plot(x, vT_true_plot, label="Ground Truth v(x, T)", color='blue')
    plt.plot(x, vT_pred_plot, label="PINO Prediction v(x, T)", color='red', linestyle='--')
    plt.title("Final Velocity at t=T")
    plt.xlabel("x")
    plt.ylabel("v(x,T)")
    plt.legend()
    plt.grid(True, linestyle='--')

    # Plot 4: Position Error
    plt.subplot(2, 2, 4)
    error_u = torch.abs(uT_true_plot - uT_pred_plot)
    error_v = torch.abs(vT_true_plot - vT_pred_plot)
    plt.plot(x, error_u, label="Position Error |u_true - u_pred|", color='green')
    plt.plot(x, error_v, label="Velocity Error |v_true - v_pred|", color='orange', linestyle='--')
    plt.title("Prediction Error at t=T")
    plt.xlabel("x")
    plt.ylabel("Absolute Error")
    plt.legend()
    plt.grid(True, linestyle='--')

    plt.tight_layout()
    plt.show()


print("\nVisualizing results on a new, unseen initial condition...")
visualize_results(trained_model, test_ic, test_sol)

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='Total Loss', color='black', linewidth=2)
plt.plot(data_loss_history, label='Data Loss', color='blue', alpha=0.7)
plt.plot(pde_loss_history, label='PDE Loss', color='red', alpha=0.7)
plt.title('Training Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




