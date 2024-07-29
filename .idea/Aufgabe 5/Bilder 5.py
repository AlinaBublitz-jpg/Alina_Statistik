import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
x = np.array([2, 4, 6, 8, 10])
y = np.array([3, 7, 8, 11, 13])

# Perform linear regression
coefficients = np.polyfit(x, y, 1)
poly = np.poly1d(coefficients)
y_hat = poly(x)

# Calculate residuals
residuals = y - y_hat

# Plotting the data points
plt.scatter(x, y, color='darkblue', zorder=5)

# Plotting the regression line
plt.plot(x, y_hat, color='gray', zorder=2, label='Regressionsgerade / Ausgleichsgerade')

# Plotting residuals and their squares
for i in range(len(x)):
    plt.vlines(x[i], y_hat[i], y[i], color='darkblue', linestyle='dotted', zorder=3)
    plt.text(x[i] + 0.1, (y[i] + y_hat[i])/2, f'{{e{i+1}}}', fontsize=10, color='darkblue', verticalalignment='bottom')
    plt.gca().add_patch(plt.Rectangle((x[i] - 0.8, min(y[i], y_hat[i])), 1.6, abs(y[i] - y_hat[i]), edgecolor='darkblue', facecolor='blue', alpha=0.2, zorder=1))

# Labeling the axes
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# Adding the title
plt.title('Punktwolke, Regressionsgerade und Residuenquadrate')

# Display the plot
plt.grid(True)
plt.show()


import matplotlib.pyplot as plt
import numpy as np

def draw_target(ax, points, title, label_left, label_right=None):
    # Draw concentric circles
    for radius, color in zip([3, 2, 1], ['lightblue', 'white', 'red']):
        circle = plt.Circle((0, 0), radius, color=color, alpha=0.5)
        ax.add_artist(circle)

    # Plot points
    ax.plot(points[:, 0], points[:, 1], 'bo')

    # Set limits and title
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal', 'box')
    ax.set_title(title)
    ax.axis('off')

    # Add left label if needed
    if label_left:
        ax.annotate(label_left, xy=(-4, 0), xytext=(-4, 0), fontsize=12, color='black', va='center', ha='center', rotation=90)

    # Add right label if needed
    if label_right:
        ax.annotate(label_right, xy=(4, 0), xytext=(4, 0), fontsize=12, color='black', va='center', ha='center', rotation=90)

# Generate points for each quadrant
np.random.seed(0)
points_low_bias_low_variance = np.random.normal(-0.5, 0.2, (30, 2)) + np.array([0, 0.5])  # Points in the left half of the red circle, shifted down
points_low_bias_high_variance = np.random.normal(-0.5, 0.7, (30, 2))  # Points spread mostly to the left, within the second ring
points_high_bias_low_variance = np.random.normal(-0.5, 0.2, (30, 2)) + np.array([0.5, 1.5])  # Points tightly clustered, shifted right and up more
points_high_bias_high_variance = np.random.normal(-0.5, 0.5, (30, 2)) + np.array([0, 2])  # Points spread out less, shifted up more

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

draw_target(axs[0, 0], points_low_bias_low_variance, '', 'Niedrige Bias', 'Niedrige Bias')
draw_target(axs[0, 1], points_low_bias_high_variance, '', None, 'Hohe Varianz')
draw_target(axs[1, 0], points_high_bias_low_variance, '', 'Hohe Bias')
draw_target(axs[1, 1], points_high_bias_high_variance, '', None)

# Add column labels
fig.text(0.38, 0.92, 'Niedrige Varianz', ha='right', fontsize=12)
fig.text(0.72, 0.92, 'Hohe Varianz', ha='left', fontsize=12)

# Add row labels
fig.text(0.05, 0.75, 'Neidrige Bias', va='center', rotation='vertical', fontsize=12)
fig.text(0.05, 0.25, 'Hohe Bias', va='center', rotation='vertical', fontsize=12)

# Add main title
fig.suptitle('Beziehung zwischen Varianz und Bias', fontsize=16)
plt.tight_layout(rect=[0.1, 0, 1, 0.93])
plt.show()



import matplotlib.pyplot as plt
import numpy as np

# Sample data points with more scatter
np.random.seed(0)
weight = np.arange(1, 11, 1)
size = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) + np.random.normal(0, 1, 10)

# Create the plot
plt.figure(figsize=(8, 6))
plt.scatter(weight, size, color='blue', s=300)  # Larger blue points

# Customize the axes
plt.xticks(range(1, 12, 1), labels=[''] * 11)  # Hide x-axis labels
plt.yticks(range(1, 12, 1), labels=[''] * 11)  # Hide y-axis labels
plt.xlabel('Weight', fontsize=16, labelpad=20)
plt.ylabel('Size', fontsize=16, labelpad=20)
plt.gca().spines['left'].set_color('black')
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['left'].set_linewidth(4)
plt.gca().spines['bottom'].set_linewidth(4)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')

# Customize the tick marks
plt.tick_params(axis='x', colors='black', width=4, length=10)
plt.tick_params(axis='y', colors='black', width=4, length=10)
plt.gca().xaxis.set_ticks_position('bottom')
plt.gca().yaxis.set_ticks_position('left')

# Remove grid lines
plt.grid(False)

# Add the main title
plt.title('Since these data look relatively linear, we will use Linear Regression, AKA Least Squares, to model the relationship between Weight and Size.', fontsize=12, pad=20)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()






import matplotlib.pyplot as plt
import numpy as np

# Sample data points with more scatter
np.random.seed(0)
weight = np.arange(1, 11, 1)
size = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) + np.random.normal(0, 1, 10)

# Perform linear regression
coefficients = np.polyfit(weight, size, 1)
poly = np.poly1d(coefficients)
size_hat = poly(weight)

# Create the plot
plt.figure(figsize=(8, 6))
plt.scatter(weight, size, color='blue', s=300)  # Larger blue points
plt.plot(weight, size_hat, color='black')  # Regression line

# Add vertical dashed lines to represent residuals
for i in range(len(weight)):
    plt.plot([weight[i], weight[i]], [size[i], size_hat[i]], 'r--')

# Customize the axes
plt.xticks(range(1, 12, 1), labels=[''] * 11)  # Hide x-axis labels
plt.yticks(range(1, 12, 1), labels=[''] * 11)  # Hide y-axis labels
plt.xlabel('Weight', fontsize=16, labelpad=20)
plt.ylabel('Size', fontsize=16, labelpad=20)
plt.gca().spines['left'].set_color('black')
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['left'].set_linewidth(4)
plt.gca().spines['bottom'].set_linewidth(4)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')

# Customize the tick marks
plt.tick_params(axis='x', colors='black', width=4, length=10)
plt.tick_params(axis='y', colors='black', width=4, length=10)
plt.gca().xaxis.set_ticks_position('bottom')
plt.gca().yaxis.set_ticks_position('left')

# Remove grid lines
plt.grid(False)

# Add the main title
plt.title('In other words, we find the line that results in the minimum sum of squared residuals.', fontsize=12, pad=20)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# Ausgabe der Designmatrix in der Konsole
print("Designmatrix (Vandermonde-Matrix) für polynomiales Modell bis Grad 12:")
print(X_df)
