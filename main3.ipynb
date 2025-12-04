import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

# Check for GPU and set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


# --- PINO Model Definition: Fourier Neural Operator (FNO) ---

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
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()
        self.modes1 = modes
        self.width = width

        self.fc0 = nn.Linear(2, self.width)  # input: [u0(x), x]

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)  # output: u(x, t=1)

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


# --- Data Generation for Heat Equation ---

def heat_equation_fd_solver(u0, nx, nt, alpha, dt):
    """Solves the 1D heat equation: u_t = alpha * u_xx"""
    dx = 2.0 / (nx - 1)
    u = u0.clone()

    for _ in range(nt):
        un = u.clone()
        # Central difference for u_xx
        u_xx = (un[2:] - 2 * un[1:-1] + un[:-2]) / (dx ** 2)
        # Update using forward Euler
        u[1:-1] = un[1:-1] + alpha * dt * u_xx
        # Boundary conditions (Dirichlet)
        u[0] = 0.0
        u[-1] = 0.0

    return u


def generate_heat_equation_data(num_samples, nx, nt, alpha=0.01, total_time=1.0):
    """Generates data for heat equation: u_t = alpha * u_xx"""
    x = torch.linspace(-1, 1, nx)
    dt = total_time / nt

    # Generate diverse initial conditions using sine waves
    u0_data = torch.zeros(num_samples, nx)

    for i in range(num_samples):
        # Create initial condition with multiple sine waves
        num_waves = torch.randint(1, 4, (1,)).item()
        u0 = torch.zeros_like(x)

        for j in range(num_waves):
            k = torch.randint(1, 6, (1,)).item()  # wave number
            amplitude = 0.5 + 0.5 * torch.rand(1).item()  # amplitude between 0.5-1.0
            phase = 2 * np.pi * torch.rand(1).item()  # random phase
            u0 += amplitude * torch.sin(k * np.pi * x + phase)

        # Normalize and ensure boundaries are zero
        u0 = (u0 - u0[0]) * (1.0 - torch.abs(x))  # enforce zero boundaries
        u0_data[i] = u0

    # Solve heat equation for each initial condition
    u_final = torch.zeros_like(u0_data)

    for i in range(num_samples):
        u_final[i] = heat_equation_fd_solver(u0_data[i], nx, nt, alpha, dt)

    # Add spatial coordinate as a channel
    initial_conditions = torch.stack([u0_data, x.repeat(num_samples, 1)], dim=-1)

    return initial_conditions.to(device), u_final.unsqueeze(-1).to(device)


# --- Training the PINO model ---

def train_pino(model, train_loader, epochs=500, pde_weight=0.1):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

    loss_history = []
    data_loss_history = []
    pde_loss_history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        data_loss_total = 0.0
        pde_loss_total = 0.0
        batch_count = 0

        for initial_conditions, final_solutions in train_loader:
            optimizer.zero_grad()

            # 1. Data Loss (Supervised)
            pred_final = model(initial_conditions)
            loss_data = F.mse_loss(pred_final, final_solutions)

            # 2. Physics Loss (Heat Equation: u_t = alpha * u_xx)
            u_pred = pred_final.squeeze(-1)
            u_initial = initial_conditions[..., 0]

            # Time derivative approximation (u_t ≈ (u_final - u_initial)/Δt)
            dt = 1.0  # total time is 1.0
            u_t_approx = (u_pred - u_initial) / dt

            # Spatial derivatives using finite differences
            nx = initial_conditions.shape[1]
            dx = 2.0 / (nx - 1)

            # Central difference for u_xx
            u_xx = (u_pred[:, 2:] - 2 * u_pred[:, 1:-1] + u_pred[:, :-2]) / (dx ** 2)

            # Trim u_t to match u_xx shape
            u_t_trimmed = u_t_approx[:, 1:-1]

            # Heat equation: u_t = alpha * u_xx
            alpha = 0.01  # thermal diffusivity
            pde_residual = u_t_trimmed - alpha * u_xx
            loss_pde = F.mse_loss(pde_residual, torch.zeros_like(pde_residual))

            # Total loss
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
NX = 128  # Reduced resolution for faster training
NT = 500  # Reduced time steps
BATCH_SIZE = 16

print("Generating training data for Heat Equation...")
train_ic, train_sol = generate_heat_equation_data(N_TRAIN, NX, NT)
print(f"Training data shapes: {train_ic.shape}, {train_sol.shape}")
print(f"Solution range: [{torch.min(train_sol):.3f}, {torch.max(train_sol):.3f}]")

print("Generating test data...")
test_ic, test_sol = generate_heat_equation_data(N_TEST, NX, NT)

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

        pred_solution = model(initial_cond).squeeze()

    # Detach and move to CPU for plotting
    x = torch.linspace(-1, 1, initial_cond.shape[1])
    initial_cond_plot = initial_cond.squeeze()[..., 0].cpu()
    true_solution_plot = true_solution.squeeze().cpu()
    pred_solution_plot = pred_solution.cpu()

    print(f"True solution range: [{torch.min(true_solution_plot):.3f}, {torch.max(true_solution_plot):.3f}]")
    print(f"Pred solution range: [{torch.min(pred_solution_plot):.3f}, {torch.max(pred_solution_plot):.3f}]")

    plt.figure(figsize=(18, 5))

    # Plot 1: Initial Condition
    plt.subplot(1, 3, 1)
    plt.plot(x, initial_cond_plot, label="Initial Condition u(x, 0)", color='black')
    plt.title("Input to the Operator")
    plt.xlabel("x")
    plt.ylabel("u(x,0)")
    plt.legend()
    plt.grid(True, linestyle='--')

    # Plot 2: True vs. Predicted Solution
    plt.subplot(1, 3, 2)
    plt.plot(x, true_solution_plot, label="Ground Truth u(x, 1)", color='blue')
    plt.plot(x, pred_solution_plot, label="PINO Prediction u(x, 1)", color='red', linestyle='--')
    plt.title("Heat Equation Solution at t=1")
    plt.xlabel("x")
    plt.ylabel("u(x,1)")
    plt.legend()
    plt.grid(True, linestyle='--')

    # Plot 3: Error
    plt.subplot(1, 3, 3)
    error = torch.abs(true_solution_plot - pred_solution_plot)
    plt.plot(x, error, label="Absolute Error", color='green')
    plt.title("Prediction Error")
    plt.xlabel("x")
    plt.ylabel("|True - Predicted|")
    plt.legend()
    plt.grid(True, linestyle='--')

    plt.tight_layout()
    plt.show()


print("\nVisualizing results on a new, unseen initial condition...")
visualize_results(trained_model, test_ic, test_sol)

# Plot training history
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(loss_history, label='Total Loss', color='black', linewidth=2)
plt.plot(data_loss_history, label='Data Loss', color='blue', alpha=0.7)
plt.plot(pde_loss_history, label='PDE Loss', color='red', alpha=0.7)
plt.title('Training Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(loss_history, label='Total Loss', color='black', linewidth=2)
plt.title('Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# Test on multiple samples
def test_model(model, test_ic, test_sol):
    model.eval()
    with torch.no_grad():
        mse_errors = []
        for i in range(min(5, len(test_ic))):
            initial_cond = test_ic[i:i + 1]
            true_solution = test_sol[i]
            pred_solution = model(initial_cond).squeeze()

            mse = F.mse_loss(pred_solution, true_solution.squeeze())
            mse_errors.append(mse.item())

            print(f"Sample {i}: MSE = {mse.item():.6f}")

        print(f"Average MSE: {np.mean(mse_errors):.6f}")


print("\nTesting on multiple samples:")
test_model(trained_model, test_ic, test_sol)