import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Constants
k = 1  # Coulomb's constant (arbitrary units)
grid_size = 10.0  # Changed to float
grid_resolution = 50 #streamlit has to come back here every time, so this being larger adds up 
x = np.linspace(-grid_size, grid_size, grid_resolution)
y = np.linspace(-grid_size, grid_size, grid_resolution)
X, Y = np.meshgrid(x, y)

# Initialize session state for charges if it doesn't exist
if 'charges' not in st.session_state:
    st.session_state.charges = []
if 'adding_charge' not in st.session_state:
    st.session_state.adding_charge = False
if 'temp_coords' not in st.session_state:
    st.session_state.temp_coords = (0.0, 0.0)
if 'selected_charge' not in st.session_state:
    st.session_state.selected_charge = None
if 'moving_charge' not in st.session_state:
    st.session_state.moving_charge = False

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(-grid_size, grid_size)
ax.set_ylim(-grid_size, grid_size)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Electric Field and Potential')

# Calculate potential and field
V = np.zeros_like(X)
Ex = np.zeros_like(X)
Ey = np.zeros_like(X)

for x0, y0, q in st.session_state.charges:
    R = np.sqrt((X - x0)**2 + (Y - y0)**2)
    V += k * q / np.maximum(R, 0.1)
    Ex += k * q * (X - x0) / (R**3 + 1e-10)
    Ey += k * q * (Y - y0) / (R**3 + 1e-10)

# Calculate reference potential at a fixed distance from charges
reference_distance = 1.0  # Distance at which to evaluate potential
reference_potential = 0.0
for x0, y0, q in st.session_state.charges:
    reference_potential += k * q / reference_distance #ensures darkest color goes next to strongest charge

# Use the reference potential to scale the visualization
vmax = np.max(abs(reference_potential)) if len(st.session_state.charges) > 0 else 1.0

# Create the potential heatmap with fixed scaling
heatmap = ax.imshow(V, extent=[-grid_size, grid_size, -grid_size, grid_size],
                    origin='lower', cmap='RdBu_r', alpha=0.7,  # Increased alpha
                    vmin=-vmax, vmax=vmax)
plt.colorbar(heatmap, ax=ax, label='Electric Potential')

# Add equipotential lines
if len(st.session_state.charges) > 0:
    # Calculate number of contour lines based on the number of charges
    n_contours = min(16, 4 * len(st.session_state.charges))
    levels = np.linspace(-vmax, vmax, n_contours)
    contours = ax.contour(X, Y, V, levels=levels, colors='black', alpha=0.3, linewidths=1.5, linestyles='solid')

# Create electric field arrows with logarithmic scaling
skip = 4  # Plot every 4th arrow
arrow_positions = np.array([(x[i], y[j]) for i in range(0, len(x), skip)
                           for j in range(0, len(y), skip)])

# Calculate field magnitude and normalize logarithmically
E_mag = np.sqrt(Ex**2 + Ey**2)
max_mag = np.max(E_mag)

# Only show field arrows if there are charges
if len(st.session_state.charges) > 0 and max_mag > 0:
    # Find minimum non-zero magnitude
    non_zero_mag = E_mag[E_mag > 0]
    if len(non_zero_mag) > 0:
        min_mag = np.min(non_zero_mag)
        
        # Logarithmic scaling for arrow lengths
        log_mag = np.log10(E_mag + 1e-10)  # Add small constant to handle zeros
        log_min = np.log10(min_mag)
        log_max = np.log10(max_mag)
        normalized_mag = (log_mag - log_min) / (log_max - log_min)
    else:
        normalized_mag = np.ones_like(E_mag)

    # Normalize field components
    Ex_norm = Ex / (E_mag + 1e-10)
    Ey_norm = Ey / (E_mag + 1e-10)

    # Create arrays in the correct order for quiver
    X_skip = X[::skip, ::skip]
    Y_skip = Y[::skip, ::skip]
    Ex_skip = Ex_norm[::skip, ::skip]
    Ey_skip = Ey_norm[::skip, ::skip]
    mag_skip = normalized_mag[::skip, ::skip]

    # Scale arrows by normalized magnitude
    arrows = ax.quiver(X_skip.flatten(), Y_skip.flatten(),
                      Ex_skip.flatten(), Ey_skip.flatten(),
                      mag_skip.flatten(),
                      color='black', alpha=0.6,
                      cmap='gray', clim=[0, 1])

# Add charge markers and labels
for x0, y0, q in st.session_state.charges:
    # Add charge circle with reduced alpha
    color = 'red' if q > 0 else 'blue'
    circle = Circle((x0, y0), 0.5, color=color, zorder=5, alpha=0.7)
    ax.add_patch(circle)
    
    # Add charge label
    label = f'{q:.1f}'
    ax.text(x0, y0 + 0.8, label, 
            horizontalalignment='center',
            verticalalignment='center',
            color='black',
            fontweight='bold',
            zorder=6)

# Calculate and display forces on charges
if len(st.session_state.charges) > 0:
    # Calculate forces on each charge
    force_x = np.zeros(len(st.session_state.charges))
    force_y = np.zeros(len(st.session_state.charges))
    
    for i, (x0, y0, q0) in enumerate(st.session_state.charges):
        # Get E-field at charge position by summing contributions from other charges
        for j, (x1, y1, q1) in enumerate(st.session_state.charges):
            if i != j:  # Skip self-interaction
                R = np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
                force_x[i] += -k * q0 * q1 * (x1 - x0) / (R**3 + 1e-10)
                force_y[i] += -k * q0 * q1 * (y1 - y0) / (R**3 + 1e-10)
    
    # Normalize forces for visualization
    force_mag = np.sqrt(force_x**2 + force_y**2)
    max_force = np.max(force_mag)
    if max_force > 0:  # Avoid division by zero
        force_x = force_x / max_force * 2  # Scale factor of 2 for visibility
        force_y = force_y / max_force * 2
    
    # Plot force arrows on charges
    for i, (x0, y0, q) in enumerate(st.session_state.charges):
        if force_mag[i] > 0:  # Only show arrows if there's a non-zero force
            ax.arrow(x0, y0, force_x[i], force_y[i], 
                    color='black', width=0.1, head_width=0.3,
                    zorder=7)  # Higher zorder to ensure arrows are visible

# Controls
st.sidebar.title('Controls')

# Charge value slider
charge_value = st.sidebar.slider('Charge Value', -5.0, 5.0, 1.0, 0.1)

# Add charge section
st.sidebar.subheader('Add New Charge')
if not st.session_state.adding_charge:
    if st.sidebar.button('Start Adding Charge'):
        st.session_state.adding_charge = True
        st.session_state.moving_charge = False
        st.rerun()
else:
    # Get coordinates
    click_x = st.sidebar.number_input('X coordinate', -grid_size, grid_size, 
                                    st.session_state.temp_coords[0], 0.1, key='x_coord')
    click_y = st.sidebar.number_input('Y coordinate', -grid_size, grid_size, 
                                    st.session_state.temp_coords[1], 0.1, key='y_coord')
    
    # Update temporary coordinates
    st.session_state.temp_coords = (click_x, click_y)
    
    # Show preview charge
    preview_circle = Circle((click_x, click_y), 0.5, color='green', zorder=5, alpha=0.5)
    ax.add_patch(preview_circle)
    st.pyplot(fig)
    
    col1, col2 = st.sidebar.columns(2)
    if col1.button('Confirm'):
        st.session_state.charges.append((click_x, click_y, charge_value))
        st.session_state.adding_charge = False
        st.session_state.temp_coords = (0.0, 0.0)
        st.rerun()
    if col2.button('Cancel'):
        st.session_state.adding_charge = False
        st.session_state.temp_coords = (0.0, 0.0)
        st.rerun()

# Move charge section
st.sidebar.subheader('Move Charge')
if not st.session_state.moving_charge:
    if len(st.session_state.charges) > 0:
        # Create a dropdown to select which charge to move
        charge_options = [f'Charge {i+1}: ({x:.1f}, {y:.1f}) = {q:.1f}' 
                         for i, (x, y, q) in enumerate(st.session_state.charges)]
        selected = st.sidebar.selectbox('Select charge to move:', charge_options)
        if st.sidebar.button('Start Moving Charge'):
            st.session_state.selected_charge = charge_options.index(selected)
            st.session_state.moving_charge = True
            st.session_state.adding_charge = False
            st.rerun()
else:
    # Get new coordinates
    old_x, old_y, old_q = st.session_state.charges[st.session_state.selected_charge]
    new_x = st.sidebar.number_input('New X coordinate', -grid_size, grid_size, 
                                   old_x, 0.1, key='move_x')
    new_y = st.sidebar.number_input('New Y coordinate', -grid_size, grid_size, 
                                   old_y, 0.1, key='move_y')
    
    # Show preview of new position
    preview_circle = Circle((new_x, new_y), 0.5, color='yellow', zorder=5, alpha=0.5)
    ax.add_patch(preview_circle)
    st.pyplot(fig)
    
    col1, col2 = st.sidebar.columns(2)
    if col1.button('Confirm Move'):
        # Update the charge position
        st.session_state.charges[st.session_state.selected_charge] = (new_x, new_y, old_q)
        st.session_state.moving_charge = False
        st.session_state.selected_charge = None
        st.rerun()
    if col2.button('Cancel Move'):
        st.session_state.moving_charge = False
        st.session_state.selected_charge = None
        st.rerun()

# Remove last charge button
if st.sidebar.button('Remove Last Charge'):
    if st.session_state.charges:
        st.session_state.charges.pop()
        st.rerun()

# Clear all charges button
if st.sidebar.button('Clear All Charges'):
    st.session_state.charges = []
    st.rerun()

# Display current charges
st.sidebar.title('Current Charges')
for i, (x, y, q) in enumerate(st.session_state.charges):
    st.sidebar.write(f'Charge {i+1}: ({x:.1f}, {y:.1f}) = {q:.1f}')

# Display the main plot only when not adding or moving a charge
if not (st.session_state.adding_charge or st.session_state.moving_charge):
    st.pyplot(fig) 