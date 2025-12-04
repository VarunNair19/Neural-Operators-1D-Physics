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

# Define the *range* of parameters we will train on
# We're re-framing the problem to learn the map from 'k' to u(x),
# where k = q / EA. This is a much more stable feature.

# Calculate the min/max range for k = q / EA
# Min k (most negative): q_min / EA_min = -100 / 1e6 = -1e-4
# Max k (most positive): q_max / EA_min =  100 / 1e6 =  1e-4
K_RANGE = [-1e-4, 1e-4]

# Keep the original ranges for user input checks
EA_RANGE = [1e6, 1e8]  # N
Q_RANGE = [-100.0, 100.0]  # N/m


# --- 2. Define the PINO Model (Parametric PINN) ---
class ParametricPINN(nn.Module):
    """
    This is a Neural Operator.
    It learns the map: G(x, k) -> u(x), where k = q/EA

    Input: a 2-vector [x, k_norm]
    Output: a 1-vector [u(x)]
    """

    def __init__(self):
        super(ParametricPINN, self).__init__()
        # Input layer takes 2 features: x, k (normalized)
        self.net = nn.Sequential(
            nn.Linear(2, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 1)  # Output is a single value, u(x)
        )

    def forward(self, x_params):
        return self.net(x_params)


# --- 3. Normalization ---
def normalize_k(k):
    """Normalizes k to the range [-1, 1]"""
    k_norm = 2.0 * (k - K_RANGE[0]) / (K_RANGE[1] - K_RANGE[0]) - 1.0
    return k_norm


def get_analytical_solution(x, EA, q, L):
    """The exact solution for this problem"""
    return (q / EA) * (L * x - (x ** 2) / 2.0)


# --- 4. The Physics-Informed Loss Function ---
def compute_operator_loss(model, batch_size):
    """
    This is the core of the PINO.
    It computes the loss over a BATCH of random problems.
    """

    # --- 1. Sample Random Problems ---
    # Generate a batch of random 'k' values
    k_batch = torch.rand(batch_size, 1, device=device) * (K_RANGE[1] - K_RANGE[0]) + K_RANGE[0]

    # Normalize them for the network input
    k_norm_batch = normalize_k(k_batch)

    # --- 2. PDE Loss (Physics Residual) ---
    # Sample random collocation points 'x' for the PDE
    n_pde_points = 100
    x_pde = torch.rand(batch_size * n_pde_points, 1, device=device) * L
    x_pde.requires_grad = True

    # Tile the parameters to match the x points
    k_pde = k_batch.repeat(n_pde_points, 1)
    k_norm_pde = k_norm_batch.repeat(n_pde_points, 1)

    # Combine inputs: [x, k_norm]
    pde_input = torch.cat([x_pde, k_norm_pde], dim=1)

    # Get model prediction
    u = model(pde_input)

    # Compute derivatives u' and u'' w.r.t. x
    u_x = torch.autograd.grad(u, x_pde, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_pde, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    # Compute the *normalized* PDE residual: R = u'' + k
    residual_pde = u_xx + k_pde
    loss_pde = torch.mean(residual_pde ** 2)

    # --- 3. Boundary Condition (BC) Loss ---

    # BC 1: Fixed end at x=0 (Dirichlet BC)
    # We want u(0, k) = 0 for all k
    x_bc_0 = torch.zeros_like(k_batch, requires_grad=True)
    bc_0_input = torch.cat([x_bc_0, k_norm_batch], dim=1)
    u_bc_0 = model(bc_0_input)
    loss_bc_0 = torch.mean(u_bc_0 ** 2)

    # BC 2: Free end at x=L (Neumann BC)
    # We want u'(L, k) = 0 for all k
    x_bc_L = torch.full_like(k_batch, L, requires_grad=True)
    bc_L_input = torch.cat([x_bc_L, k_norm_batch], dim=1)
    u_bc_L = model(bc_L_input)

    # Get u'(L)
    u_x_bc_L = torch.autograd.grad(u_bc_L, x_bc_L, grad_outputs=torch.ones_like(u_bc_L), create_graph=True)[0]

    # The loss is just the square of the slope
    loss_bc_L = torch.mean(u_x_bc_L ** 2)

    # --- 4. Total Loss ---
    # We weight the BC losses higher because they are critical
    # Now that all terms are small, the weighting is more effective
    total_loss = loss_pde + 100.0 * loss_bc_0 + 100.0 * loss_bc_L

    return total_loss


# --- 5. Training Function ---
def train_operator(epochs=5000, batch_size=32):
    print(f"Starting training on {device} for {epochs} epochs...")
    model = ParametricPINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    start_time = datetime.datetime.now()  # Use datetime.datetime
    loss_history = []  # To store loss values

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = compute_operator_loss(model, batch_size)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())  # Store loss

        if (epoch + 1) % 500 == 0:
            # --- LOSS FORMATTING FIX (Change 3) ---
            # Print in scientific notation to see small numbers
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4e}")

    end_time = datetime.datetime.now()  # Use datetime.datetime
    print(f"Training complete. Time taken: {end_time - start_time}")
    return model, loss_history  # Return history


# --- 6. New Function: Plot Training Loss ---
def plot_training_loss(loss_history):
    """
    Plots the loss curve to show how the operator learns.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Total Loss')
    plt.title('Training Loss History (How the Operator Learns)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.yscale('log')  # Use log scale to see the drop
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.show()


# --- 7. Prediction & Plotting Function (Modified) ---
def predict_and_plot(model, material_name, E, A, q_load):
    """
    Uses the *trained* operator to predict the solution for a *new* problem.
    """
    print(f"\n--- Predicting for: {material_name} ---")
    model.eval()  # Set model to evaluation mode

    EA = E * A
    k = q_load / EA

    # Check if parameters are in the trained range
    if not (EA_RANGE[0] <= EA <= EA_RANGE[1]):
        print(f"Warning: EA={EA:.2e} is outside the trained range [{EA_RANGE[0]:.2e}, {EA_RANGE[1]:.2e}]")
    if not (Q_RANGE[0] <= q_load <= Q_RANGE[1]):
        print(f"Warning: q={q_load:.2f} is outside the trained range [{Q_RANGE[0]:.2f}, {Q_RANGE[1]:.2f}]")
    if not (K_RANGE[0] <= k <= K_RANGE[1]):
        print(f"Warning: k={k:.2e} is outside the trained k_range [{K_RANGE[0]:.2e}, {K_RANGE[1]:.2e}]")

    # Normalize the new problem's parameter k
    k_norm = normalize_k(torch.tensor(k))

    # Create the input tensor for plotting
    n_plot = 101
    x_plot = torch.linspace(0, L, n_plot, device=device).view(-1, 1)

    # Tile the k parameter to match the x_plot
    k_norm_plot = torch.full_like(x_plot, k_norm.item())

    # Combine into the model's input format [x, k_norm]
    plot_input = torch.cat([x_plot, k_norm_plot], dim=1)

    # Get the prediction (NO TRAINING, just forward pass)
    with torch.no_grad():
        u_pred = model(plot_input).cpu().numpy()

    # Get the exact analytical solution for comparison
    x_np = x_plot.cpu().numpy()
    u_exact = get_analytical_solution(x_np, EA, q_load, L)

    # Calculate error
    error = np.abs(u_pred - u_exact)

    # Plot the results
    plt.figure(figsize=(12, 6))  # Changed figure size

    # Subplot 1: Displacement
    plt.subplot(1, 2, 1)
    plt.plot(x_np, u_exact, 'b-', label='Analytical Solution', linewidth=2)
    plt.plot(x_np, u_pred, 'r--', label='PINO Prediction', markersize=4)
    plt.title(f"PINO Prediction for {material_name}\nEA={EA:.2e} N, q={q_load:.2f} N/m")
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


# --- 8. Main Execution (Modified) ---
if __name__ == "__main__":
    # --- Train the Operator Once ---
    # This part takes time (e.g., 5-10 minutes)
    trained_operator, history = train_operator(epochs=8000, batch_size=32)

    # --- Show how the operator learned ---
    print("\nVisualizing the training process (how the operator learned)...")
    plot_training_loss(history)

    # --- Now, Predict for Different Cases Instantly ---
    # This part is immediate. We are *using* the trained operator.

    while True:
        print("\n--- PINO Prediction Mode ---")
        print(f"Enter parameters for a new bar.")
        # Print original ranges for user context
        print(f"Trained EA Range: [{EA_RANGE[0]:.2e}, {EA_RANGE[1]:.2e}] N")
        print(f"Trained q Range: [{Q_RANGE[0]:.2f}, {Q_RANGE[1]:.2f}] N/m")
        print(f"(This implies a 'k' (q/EA) range of [{K_RANGE[0]:.2e}, {K_RANGE[1]:.2e}])")

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
            predict_and_plot(trained_operator, material_name, E, A, q_load)

        except ValueError as e:
            print(f"Invalid input: {e}. Please enter numbers.")

        try_again = input("Predict for another material? (y/n): ").strip().lower()
        if try_again != 'y':
            break

    print("PINO demonstration complete.")