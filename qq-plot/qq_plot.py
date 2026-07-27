import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import norm


def get_data(kind):
    """
    Generate sample data for Q-Q plot analysis.

    This function provides datasets that can be used to evaluate
    normality with a Quantile-Quantile (Q-Q) plot. New distributions
    can easily be added by creating additional ``case`` statements.

    Parameters
    ----------
    kind : str
        Specifies which dataset to generate.

        Supported options:
        - "normal"
            Generates 20 normally distributed values with mean 50 and
            standard deviation 10, then adds additional random noise
            (mean 0, standard deviation 3).

        - "ref"
            Returns a fixed reference dataset consisting of 20 values.

    Returns
    -------
    numpy.ndarray
        One-dimensional array containing the generated dataset.

    Raises
    ------
    ValueError
        If an unsupported dataset name is provided.

    Examples
    --------
    >>> data = get_data("normal")
    >>> data = get_data("ref")
    """

    match kind:

        case "normal":
            # Set the random seed for reproducible results.
            np.random.seed(42)

            # Generate normally distributed values.
            normal_data = np.random.normal(
                loc=50,      # Mean
                scale=10,    # Standard deviation
                size=20      # Number of samples
            )

            # Generate random measurement noise.
            noise = np.random.normal(
                loc=0,
                scale=3,
                size=20
            )

            # Combine the normal data with noise.
            data = normal_data + noise

        case "ref":
            # Fixed dataset for testing and demonstrations.
            data = np.array([
                23, 18, 30, 25, 21,
                19, 28, 27, 24, 20,
                22, 26, 31, 17, 29,
                24, 22, 18, 27, 25
            ])

        # Example for adding another distribution:
        #
        # case "exponential":
        #     data = np.random.exponential(
        #         scale=10,
        #         size=20
        #     )

        case _:
            raise ValueError(f"Unknown data type: {kind}")

    return data


# =============================================================================
# Select the dataset to analyze.
#
# Change only the argument below:
#     "normal" -> randomly generated normal dataset
#     "ref"    -> fixed reference dataset
# =============================================================================
data = get_data("normal")


# =============================================================================
# Prepare data for the Q-Q plot
# =============================================================================

# Sort the observations from smallest to largest.
data_sorted = np.sort(data)

# Number of observations.
n = len(data_sorted)

# Compute plotting positions (empirical cumulative probabilities).
p = (np.arange(1, n + 1) - 0.5) / n

# Convert cumulative probabilities into theoretical normal quantiles.
z = norm.ppf(p)


# =============================================================================
# Create the Q-Q plot
# =============================================================================

# Plot ordered observations against theoretical normal quantiles.
plt.scatter(z, data_sorted)

# Compute the least-squares reference line.
slope, intercept = np.polyfit(z, data_sorted, 1)

# Plot the fitted reference line.
plt.plot(z, slope * z + intercept, color="red")

# Label the figure.
plt.xlabel("Theoretical Normal Quantiles")
plt.ylabel("Ordered Data Values")
plt.title("Q-Q Plot")

plt.grid(True)
plt.ion()
plt.show()


# =============================================================================
# Create a table containing the Q-Q plot calculations
# =============================================================================

qq_table = pd.DataFrame({
    "i": np.arange(1, n + 1),
    "ordered_point": data_sorted,
    "p_percentile": p,
    "z_value": z
})

print(qq_table)


# =============================================================================
# Export the calculation table to Excel
# =============================================================================

excel_path = "qq_plot_data.xlsx"

qq_table.to_excel(
    excel_path,
    index=False,
    sheet_name="qq_plot_data"
)

print(f"Excel saved: {excel_path}")


# =============================================================================
# Optional local development utilities
#
# Executes a local helper script if it exists. This is intended only
# for the developer's environment and can safely be removed or commented
# out when sharing this program.
# =============================================================================
try:
    exec(open(r"C:\Users\FMS4R\Documents\Github\local\additions.py").read())
except FileNotFoundError:
    pass
