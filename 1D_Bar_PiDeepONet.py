'''
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import datetime  # Import the datetime module

# Set a consistent random seed for reproducible results
torch.manual_seed(42)
np.random.seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. Define Problem Constants ---
L = 1.0  # Length of the bar (1.0 m)


# --- 2. Define the PINN Model ---
# This network learns the NON-DIMENSIONAL solution û(x)
# The problem is: û''(x) + 1 = 0, with û(0) = 0, û'(L) = 0
class PINN(nn.Module):
    """
    This is a PINN (not a PINO).
    It learns the single non-dimensional basis function û(x).

    Input: a 1-vector [x]
    Output: a 1-vector [û(x)]
    """

    def __init__(self):
        super(PINN, self).__init__()
        # Input layer takes 1 feature: x
        self.net = nn.Sequential(
            nn.Linear(1, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 1)  # Output is a single value, û(x)
        )

    def forward(self, x):
        return self.net(x)


# --- 3. The Physics-Informed Loss Function ---
def compute_loss(model, L):
    """
    This is the core of the PINN.
    It computes the loss for the non-dimensional problem.
    """

    # --- 1. PDE Loss (Physics Residual) ---
    n_pde_points = 100
    x_pde = torch.rand(n_pde_points, 1, device=device) * L
    x_pde.requires_grad = True

    # Get model prediction
    u_hat = model(x_pde)

    # Compute derivatives û' and û'' w.r.t. x
    u_hat_x = torch.autograd.grad(u_hat, x_pde, grad_outputs=torch.ones_like(u_hat), create_graph=True)[0]
    u_hat_xx = torch.autograd.grad(u_hat_x, x_pde, grad_outputs=torch.ones_like(u_hat_x), create_graph=True)[0]

    # Compute the non-dimensional PDE residual: R = û'' + 1
    residual_pde = u_hat_xx + 1.0
    loss_pde = torch.mean(residual_pde ** 2)

    # --- 2. Boundary Condition (BC) Loss ---

    # BC 1: Fixed end at x=0 (Dirichlet BC)
    # We want û(0) = 0
    x_bc_0 = torch.tensor([[0.0]], device=device, requires_grad=True)
    u_hat_bc_0 = model(x_bc_0)
    loss_bc_0 = torch.mean(u_hat_bc_0 ** 2)

    # BC 2: Free end at x=L (Neumann BC)
    # We want û'(L) = 0
    x_bc_L = torch.tensor([[L]], device=device, requires_grad=True)
    u_hat_bc_L = model(x_bc_L)

    # Get û'(L)
    u_hat_x_bc_L = torch.autograd.grad(u_hat_bc_L, x_bc_L, grad_outputs=torch.ones_like(u_hat_bc_L), create_graph=True)[
        0]

    # The loss is just the square of the slope
    loss_bc_L = torch.mean(u_hat_x_bc_L ** 2)

    # --- 3. Total Loss ---
    total_loss = loss_pde + 100.0 * loss_bc_0 + 100.0 * loss_bc_L

    return total_loss


# --- 4. Training Function ---
def train_pinn_basis_solver(epochs=3000, batch_size=32):
    """
    This function trains the PINN to find the single basis solution û(x).
    """
    print(f"Starting training on {device} for {epochs} epochs...")
    print("Goal: Find the non-dimensional basis solution û(x) such that û'' + 1 = 0.")
    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    start_time = datetime.datetime.now()  # Use datetime.datetime
    loss_history = []  # To store loss values

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = compute_loss(model, L)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())  # Store loss

        if (epoch + 1) % 500 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4e}")

    end_time = datetime.datetime.now()  # Use datetime.datetime
    print(f"Training complete. Time taken: {end_time - start_time}")
    return model, loss_history  # Return history


# --- 5. Function: Plot Training Loss ---
def plot_training_loss(loss_history):
    """
    Plots the loss curve to show how the operator learns.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Total Loss')
    plt.title('Training Loss History (Learning the Basis Solution û(x))')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.yscale('log')  # Use log scale to see the drop
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.show()


# --- 6. Prediction & Plotting Function (The "Operator" Step) ---
def predict_and_plot(pinn_model, material_name, E, A, q_load):
    """
    Uses the trained PINN model (û) to solve for any new problem.
    The final solution is u(x) = (q/EA) * û(x)
    """
    print(f"\n--- Predicting for: {material_name} ---")
    pinn_model.eval()  # Set model to evaluation mode

    EA = E * A
    k = q_load / EA

    print(f"Calculated k = q/EA = {q_load:.2f} / {EA:.2e} = {k:.2e}")

    # Create the input tensor for plotting
    n_plot = 101
    x_plot_tensor = torch.linspace(0, L, n_plot, device=device).view(-1, 1)

    # Get the non-dimensional basis solution û(x) from the PINN
    with torch.no_grad():
        u_hat_pred = pinn_model(x_plot_tensor).cpu().numpy()

    # --- This is the "Operator" step ---
    # Scale the basis solution by 'k' to get the final answer
    u_pred = k * u_hat_pred

    # Get the exact analytical solution for comparison
    x_np = x_plot_tensor.cpu().numpy()
    u_exact = get_analytical_solution(x_np, EA, q_load, L)

    # Calculate error
    error = np.abs(u_pred - u_exact)

    # Plot the results
    plt.figure(figsize=(12, 6))  # Changed figure size

    # Subplot 1: Displacement
    plt.subplot(1, 2, 1)
    plt.plot(x_np, u_exact, 'b-', label='Analytical Solution', linewidth=2)
    plt.plot(x_np, u_pred, 'r--', label='PINN-Operator Prediction', markersize=4)
    plt.title(f"PINN-Operator Prediction for {material_name}\nEA={EA:.2e} N, q={q_load:.2f} N/m")
    plt.xlabel("Position (x)")
    plt.ylabel("Displacement u(x)")
    plt.legend()
    plt.grid(True, linestyle=':')

    # Subplot 2: Absolute Error
    plt.subplot(1, 2, 2)
    plt.plot(x_np, error, 'k-', label='Absolute Error')
    plt.title("Prediction Accuracy (Absolute Error)")
    plt.xlabel("Position (x)")
    plt.ylabel("Error |u_pred - u_exact|")
    plt.legend()
    plt.grid(True, linestyle=':')

    plt.tight_layout()  # Add tight layout
    plt.show()


# --- 7. Analytical Solution Helper ---
def get_analytical_solution(x, EA, q, L):
    """The exact solution for this problem: u(x) = (q/EA) * (L*x - x^2/2)"""
    k = q / EA
    u_hat = (L * x - (x ** 2) / 2.0)
    return k * u_hat


# --- 8. Main Execution (Modified) ---
if __name__ == "__main__":
    # --- Train the PINN Once to find û(x) ---
    # This part takes time (e.g., 1-2 minutes)
    # We only need 3000 epochs because the problem is much simpler
    trained_pinn, history = train_pinn_basis_solver(epochs=3000, batch_size=32)

    # --- Show how the operator learned ---
    print("\nVisualizing the training process (how the basis was learned)...")
    plot_training_loss(history)

    # --- Now, Predict for Different Cases Instantly ---
    # This part is immediate. We are *using* the trained PINN.

    while True:
        print("\n--- PINN-Operator Prediction Mode ---")
        print("Enter parameters for a new bar.")

        material_name = input("Enter material name (e.g., Steel, Aluminum): ")
        if not material_name:
            material_name = "Custom Material"

        try:
            E_str = input(f"Enter Young's Modulus (E) (e.g., 70e9 for Aluminum): ")
            A_str = input(f"Enter Cross-Sectional Area (A) (e.g., 0.0001): ")
            q_str = input(f"Enter distributed load 'q' (N/m) (e.g., 50): ")

            E = float(E_str)
            A = float(A_str)
            q_load = float(q_str)

            # Use the PINO to get the solution
            predict_and_plot(trained_pinn, material_name, E, A, q_load)

        except ValueError as e:
            print(f"Invalid input: {e}. Please enter numbers.")

        try_again = input("Predict for another material? (y/n): ").strip().lower()
        if try_again != 'y':
            break

    print("PINN-Operator demonstration complete.")

'''
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

# Set a consistent random seed
torch.manual_seed(42)
np.random.seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# --- 1. Define Problem Constants ---
L = 1.0
K_RANGE = [-1e-4, 1e-4]
EA_RANGE = [1e6, 1e8]
Q_RANGE = [-100.0, 100.0]

### FIX 1: Define the output scale factor ###
# We scale u by the max k value to make the output (u_norm) O(1)
U_SCALE = K_RANGE[1]  # 1e-4


# --- 2. Define Operator Models ---

class PINO_Style_Operator(nn.Module):
    def __init__(self, layers=[2, 40, 40, 40, 40, 1]):
        super(PINO_Style_Operator, self).__init__()
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, x_params):
        return self.net(x_params)


class DeepONet_Operator(nn.Module):
    def __init__(self, p_dim=40):
        super(DeepONet_Operator, self).__init__()
        self.p_dim = p_dim
        self.branch = nn.Sequential(
            nn.Linear(1, 40), nn.Tanh(),
            nn.Linear(40, 40), nn.Tanh(),
            nn.Linear(40, self.p_dim)
        )
        self.trunk = nn.Sequential(
            nn.Linear(1, 40), nn.Tanh(),
            nn.Linear(40, 40), nn.Tanh(),
            nn.Linear(40, self.p_dim)
        )
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_params):
        x_in = x_params[:, 0:1]
        k_in = x_params[:, 1:2]
        B_out = self.branch(k_in)
        T_out = self.trunk(x_in)
        u = torch.sum(B_out * T_out, dim=1, keepdim=True) + self.b
        return u


# --- 3. Normalization & Analytical Solution ---
def normalize_k(k):
    k_norm = 2.0 * (k - K_RANGE[0]) / (K_RANGE[1] - K_RANGE[0]) - 1.0
    return k_norm


def get_analytical_solution(x, EA, q, L):
    return (q / EA) * (L * x - (x ** 2) / 2.0)


# --- 4. The Physics-Informed Loss Function (Works for BOTH models) ---
def compute_loss(model, batch_size):
    k_batch = torch.rand(batch_size, 1, device=device) * (K_RANGE[1] - K_RANGE[0]) + K_RANGE[0]
    k_norm_batch = normalize_k(k_batch)

    # --- 2. PDE Loss (Physics Residual) ---
    n_pde_points = 1000
    x_pde = torch.rand(batch_size * n_pde_points, 1, device=device) * L
    x_pde.requires_grad = True

    k_pde = k_batch.repeat(n_pde_points, 1)
    k_norm_pde = k_norm_batch.repeat(n_pde_points, 1)
    pde_input = torch.cat([x_pde, k_norm_pde], dim=1)

    # u_norm is the network's output, which we expect to be O(1)
    u_norm = model(pde_input)

    u_norm_x = torch.autograd.grad(u_norm, x_pde, grad_outputs=torch.ones_like(u_norm), create_graph=True)[0]
    u_norm_xx = torch.autograd.grad(u_norm_x, x_pde, grad_outputs=torch.ones_like(u_norm_x), create_graph=True)[0]

    ### FIX 2: Update the PDE residual with the scale factor ###
    # Original: residual_pde = u_xx + k_pde
    # New:      u_norm_xx + (k_pde / U_SCALE) = 0
    residual_pde = u_norm_xx + (k_pde / U_SCALE)
    loss_pde = torch.mean(residual_pde ** 2)

    # --- 3. Boundary Condition (BC) Loss ---
    # BCs are linear, so scaling doesn't change them.
    # u(0) = 0  => u_norm(0) * U_SCALE = 0 => u_norm(0) = 0
    # u'(L) = 0 => (u_norm(L) * U_SCALE)' = 0 => u_norm'(L) = 0

    # BC 1: u_norm(0, k) = 0
    x_bc_0 = torch.zeros_like(k_batch, requires_grad=True)
    bc_0_input = torch.cat([x_bc_0, k_norm_batch], dim=1)
    u_norm_bc_0 = model(bc_0_input)
    loss_bc_0 = torch.mean(u_norm_bc_0 ** 2)

    # BC 2: u_norm'(L, k) = 0
    x_bc_L = torch.full_like(k_batch, L, requires_grad=True)
    bc_L_input = torch.cat([x_bc_L, k_norm_batch], dim=1)
    u_norm_bc_L = model(bc_L_input)
    u_norm_x_bc_L = \
    torch.autograd.grad(u_norm_bc_L, x_bc_L, grad_outputs=torch.ones_like(u_norm_bc_L), create_graph=True)[0]
    loss_bc_L = torch.mean(u_norm_x_bc_L ** 2)

    # Loss BC code is unchanged because the target values are still 0

    total_loss = loss_pde + 100.0 * loss_bc_0 + 100.0 * loss_bc_L
    return total_loss


# --- 5. Training Function (Works for BOTH models) ---
def train_model(ModelClass, epochs=5000, batch_size=32, model_name="Model"):
    print(f"--- Starting Training for {model_name} ---")
    print(f"Training on {device} for {epochs} epochs...")

    model = ModelClass().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    start_time = time.time()
    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = compute_loss(model, batch_size)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())

        if (epoch + 1) % 1000 == 0:
            print(f"[{model_name}] Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4e}")

    end_time = time.time()
    print(f"[{model_name}] Training complete. Time taken: {end_time - start_time:.2f}s")
    return model, loss_history


# --- 6. Plot Training Loss Comparison ---
def plot_training_comparison(pino_history, don_history):
    plt.figure(figsize=(10, 6))
    plt.plot(pino_history, label='PINO-Style Operator Loss', color='blue', alpha=0.8)
    plt.plot(don_history, label='DeepONet Operator Loss', color='red', alpha=0.8)
    plt.title('Training Loss History Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.show()


# --- 7. Prediction & Plotting Function (Modified for Comparison) ---
def predict_and_compare(pino_model, don_model, material_name, E, A, q_load):
    print(f"\n--- Predicting for: {material_name} ---")
    pino_model.eval()
    don_model.eval()

    EA = E * A
    k = q_load / EA

    if not (K_RANGE[0] <= k <= K_RANGE[1]):
        print(f"Warning: k={k:.2e} is outside the trained k_range [{K_RANGE[0]:.2e}, {K_RANGE[1]:.2e}]")
        print("         Model is extrapolating, results may be inaccurate.")

    k_norm = normalize_k(torch.tensor(k))
    n_plot = 101
    x_plot = torch.linspace(0, L, n_plot, device=device).view(-1, 1)
    k_norm_plot = torch.full_like(x_plot, k_norm.item())
    plot_input = torch.cat([x_plot, k_norm_plot], dim=1)

    with torch.no_grad():
        ### FIX 3: De-normalize the model outputs ###
        # Model predicts u_norm, so we multiply by U_SCALE
        u_norm_pred_pino = pino_model(plot_input)
        u_pred_pino = u_norm_pred_pino.cpu().numpy() * U_SCALE

        u_norm_pred_don = don_model(plot_input)
        u_pred_don = u_norm_pred_don.cpu().numpy() * U_SCALE

    # Get the exact analytical solution for comparison
    x_np = x_plot.cpu().numpy()
    u_exact = get_analytical_solution(x_np, EA, q_load, L)

    # Calculate errors
    error_pino = np.abs(u_pred_pino - u_exact)
    error_don = np.abs(u_pred_don - u_exact)

    avg_err_pino = np.mean(error_pino)
    avg_err_don = np.mean(error_don)
    print(f"Average PINO Error:   {avg_err_pino:.4e}")
    print(f"Average DeepONet Error: {avg_err_don:.4e}")

    # Plot the results
    plt.figure(figsize=(18, 6))

    # Subplot 1: Displacement Comparison
    plt.subplot(1, 3, 1)
    plt.plot(x_np, u_exact, 'k-', label='Analytical Solution', linewidth=3)
    plt.plot(x_np, u_pred_pino, 'b--', label='PINO Prediction', markersize=4)
    plt.plot(x_np, u_pred_don, 'r:', label='DeepONet Prediction', markersize=4, linewidth=2)
    plt.title(f"Prediction for {material_name}\nEA={EA:.2e} N, q={q_load:.2f} N/m")
    plt.xlabel("Position (x)")
    plt.ylabel("Displacement u(x)")
    plt.legend()
    plt.grid(True, linestyle=':')

    # Subplot 2: PINO Error
    plt.subplot(1, 3, 2)
    plt.plot(x_np, error_pino, 'b-', label=f'PINO Abs Error (Avg: {avg_err_pino:.2e})')
    plt.title("PINO-Style Operator Accuracy")
    plt.xlabel("Position (x)")
    plt.ylabel("Error |u_pred - u_exact|")
    plt.legend()
    # Use log scale to see small errors
    plt.yscale('log')
    plt.grid(True, linestyle=':')

    # Subplot 3: DeepONet Error
    plt.subplot(1, 3, 3)
    plt.plot(x_np, error_don, 'r-', label=f'DeepONet Abs Error (Avg: {avg_err_don:.2e})')
    plt.title("DeepONet Operator Accuracy")
    plt.xlabel("Position (x)")
    plt.ylabel("Error |u_pred - u_exact|")
    plt.legend()
    # Use log scale to see small errors
    plt.yscale('log')
    plt.grid(True, linestyle=':')

    plt.tight_layout()
    plt.show()


# --- 8. Main Execution (Modified) ---
if __name__ == "__main__":
    ### FIX 4: Increased epochs for better convergence ###
    # Now that the loss is well-scaled, the models will
    # converge properly. 8000 epochs is a good number.
    EPOCHS = 8000

    # 1. Train PINO-Style Operator
    pino_model, pino_history = train_model(
        PINO_Style_Operator,
        epochs=EPOCHS,
        batch_size=32,
        model_name="PINO-Style"
    )

    # 2. Train DeepONet Operator
    don_model, don_history = train_model(
        DeepONet_Operator,
        epochs=EPOCHS,
        batch_size=32,
        model_name="DeepONet"
    )

    print("\nVisualizing the training process comparison...")
    plot_training_comparison(pino_history, don_history)

    # --- Interactive Prediction Loop ---
    while True:
        print("\n--- PINO vs. DeepONet Prediction Mode ---")
        print(f"Enter parameters for a new bar.")
        print(f"Trained EA Range: [{EA_RANGE[0]:.2e}, {EA_RANGE[1]:.2e}] N")
        print(f"Trained q Range: [{Q_RANGE[0]:.2f}, {Q_RANGE[1]:.2f}] N/m")

        material_name = input("Enter material name (e.g., Steel): ")
        if not material_name: material_name = "Custom Material"

        try:
            E_str = input(f"Enter Young's Modulus (E) (e.g., 200e9 for Steel): ")
            A_str = input(f"Enter Cross-Sectional Area (A) (e.g., 0.0001): ")
            q_str = input(f"Enter distributed load 'q' (N/m) (e.g., -80): ")

            E = float(E_str)
            A = float(A_str)
            q_load = float(q_str)

            predict_and_compare(pino_model, don_model,
                                material_name, E, A, q_load)

        except ValueError as e:
            print(f"Invalid input: {e}. Please enter numbers.")

        try_again = input("Predict for another material? (y/n): ").strip().lower()
        if try_again != 'y':
            break

    print("PINO/DeepONet demonstration complete.")