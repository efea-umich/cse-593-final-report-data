import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np # Import numpy for std calculation if needed

# --- Configuration ---
OUTPUT_DIR = 'plots' # Use a new directory for plots
# Set a professional Seaborn theme and context
# Context options: 'paper', 'notebook', 'talk', 'poster'
# Style options: 'darkgrid', 'whitegrid', 'dark', 'white', 'ticks'
# Palette options: 'viridis', 'plasma', 'magma', 'rocket', 'mako', 'colorblind', etc.
sns.set_theme(style="whitegrid", context="talk", palette="viridis")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.constrained_layout.use'] = True # Better spacing

# --- Helper Functions ---
def ensure_dir(directory):
    """Creates the directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

def plot_time_taken_boxplots(df, output_dir):
    """
    Generates and saves box plots for time taken per task using Seaborn.
    Uses 'user_id' as the identifier.
    """
    # --- Data Preparation for Seaborn ---
    # Melt the dataframe to long format, which is preferred by Seaborn
    time_cols = ['phone_time_taken', 'smartwatch_unassisted_time_taken', 'smartwatch_assisted_time_taken']
    # CORRECTED: Use 'user_id' instead of 'participant_id'
    melted_df = df.melt(id_vars=['user_id', 'task_group'],
                        value_vars=time_cols,
                        var_name='condition',
                        value_name='time_taken')

    # Clean up condition names for labels
    condition_map = {
        'phone_time_taken': 'Phone',
        'smartwatch_unassisted_time_taken': 'Smartwatch',
        'smartwatch_assisted_time_taken': 'Smartwatch + Intervention'
    }
    melted_df['condition'] = melted_df['condition'].map(condition_map)

    # Define the order of conditions for plotting
    condition_order = ['Phone', 'Smartwatch', 'Smartwatch + Intervention']

    # --- Plotting ---
    # Create a FacetGrid to handle plotting per task (Optional - kept for reference)
    # g = sns.catplot(
    #     data=melted_df.dropna(subset=['time_taken']), # Drop rows where time_taken is NaN
    #     x='condition',
    #     y='time_taken',
    #     col='task_group', # Create separate plots for each task group
    #     kind='box',       # Use box plot
    #     order=condition_order, # Specify the order of boxes
    #     col_order=['Task 1', 'Task 2', 'Task 3'], # Specify the order of columns (tasks)
    #     height=5,         # Height of each facet
    #     aspect=0.8,       # Aspect ratio of each facet
    #     palette="viridis", # Use a specific palette
    #     linewidth=1.5,    # Box line width
    #     fliersize=3       # Size of outlier markers
    # )
    # # --- Customization ---
    # # Set titles and labels for each subplot (Axes)
    # for ax, task in zip(g.axes.flat, ['Task 1', 'Task 2', 'Task 3']):
    #     ax.set_title(f"Time Taken: {task}", fontsize=14)
    #     ax.set_xlabel("Condition", fontsize=12)
    #     ax.set_ylabel("Time Taken (s)", fontsize=12)
    #     ax.tick_params(axis='x', rotation=15)
    # plt.close(g.fig) # Close the catplot figure if generated

    # --- Re-plotting Task 1 Separately (Alternative for strict adherence to original) ---
    # If you strictly need Task 1 to *only* show Phone and SW+Intervention:
    fig_task1, ax_task1 = plt.subplots(figsize=(6, 5))
    task1_data = melted_df[
        (melted_df['task_group'] == 'Task 1') &
        (melted_df['condition'].isin(['Phone', 'Smartwatch + Intervention']))
    ].dropna(subset=['time_taken'])

    sns.boxplot(
        data=task1_data,
        x='condition',
        y='time_taken',
        order=['Phone', 'Smartwatch + Intervention'],
        palette="viridis",
        linewidth=1.5,
        fliersize=3,
        ax=ax_task1
    )
    ax_task1.set_title("Time Taken: Task 1", fontsize=14)
    ax_task1.set_xlabel("Condition", fontsize=12)
    ax_task1.set_ylabel("Time Taken (s)", fontsize=12)
    ax_task1.tick_params(axis='x', rotation=0) # No rotation needed for 2 categories
    filename_task1 = os.path.join(output_dir, 'time_taken_boxplot_task1.png')
    fig_task1.savefig(filename_task1, bbox_inches='tight')
    print(f"Saved plot: {filename_task1}")
    plt.close(fig_task1)

    # --- Plotting Task 2 and 3 Together ---
    fig_tasks23, axes_tasks23 = plt.subplots(1, 2, figsize=(12, 5), sharey=True) # Share Y axis
    tasks_2_3 = ['Task 2', 'Task 3']
    for i, task in enumerate(tasks_2_3):
        ax = axes_tasks23[i]
        task_data = melted_df[melted_df['task_group'] == task].dropna(subset=['time_taken'])
        sns.boxplot(
            data=task_data,
            x='condition',
            y='time_taken',
            order=condition_order,
            palette="viridis",
            linewidth=1.5,
            fliersize=3,
            ax=ax
        )
        ax.set_title(f"Time Taken: {task}", fontsize=14)
        ax.set_xlabel("Condition", fontsize=12)
        ax.set_ylabel("Time Taken (s)" if i == 0 else "", fontsize=12) # Only label Y on the first plot
        ax.tick_params(axis='x', rotation=15)

    # Add a main title (optional, adjust spacing if needed)
    # fig_tasks23.suptitle('Time Taken Comparison for Tasks 2 & 3', fontsize=16, y=1.02)
    filename_tasks23 = os.path.join(output_dir, 'time_taken_boxplots_tasks2_3.png')
    # Use tight_layout or constrained_layout to adjust spacing
    fig_tasks23.tight_layout() # Or plt.subplots(constrained_layout=True)
    fig_tasks23.savefig(filename_tasks23, bbox_inches='tight')
    print(f"Saved plot: {filename_tasks23}")
    plt.close(fig_tasks23)


def plot_smartwatch_errors_barchart(df, output_dir):
    """
    Generates and saves an bar chart for smartwatch errors per 100 chars using Seaborn.
    Uses 'user_id' as the identifier.
    """
    # --- Data Preparation ---
    error_cols = ['smartwatch_unassisted_non_contiguous_errors_per_100_chars',
                  'smartwatch_assisted_non_contiguous_errors_per_100_chars']
    # CORRECTED: Use 'user_id' instead of 'participant_id'
    errors_df = df.melt(id_vars=['user_id'],
                        value_vars=error_cols,
                        var_name='condition',
                        value_name='errors_per_100_chars')

    # Clean up condition names
    condition_map = {
        'smartwatch_unassisted_non_contiguous_errors_per_100_chars': 'Unassisted',
        'smartwatch_assisted_non_contiguous_errors_per_100_chars': 'Assisted (Intervention)'
    }
    errors_df['condition'] = errors_df['condition'].map(condition_map)

    # Define order
    condition_order = ['Unassisted', 'Assisted (Intervention)']

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(7, 6)) # Adjusted figure size
    sns.barplot(
        data=errors_df.dropna(subset=['errors_per_100_chars']),
        x='condition',
        y='errors_per_100_chars',
        order=condition_order,
        palette="viridis",
        capsize=0.1,      # Size of the error bar caps
        errorbar='sd',    # Show standard deviation error bars (seaborn default is ci=95)
        ax=ax
    )

    # --- Customization ---
    ax.set_title('Smartwatch Errors per 100 Characters', fontsize=16, pad=20) # Added padding
    ax.set_xlabel('Smartwatch Condition', fontsize=12)
    ax.set_ylabel('Errors per 100 Characters', fontsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    sns.despine() # Remove top and right spines

    # --- Saving ---
    filename = os.path.join(output_dir, 'smartwatch_errors_barchart.png')
    # Use bbox_inches='tight' to prevent labels from being cut off
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved plot: {filename}")
    plt.close(fig) # Close the figure to free memory


def plot_assisted_changes_barchart(df, output_dir):
    """
    Generates and saves an bar chart for total vs erroneous changes (assisted) using Seaborn.
    Uses 'user_id' as the identifier.
    """
    # --- Data Preparation ---
    # Filter for assisted tasks with non-zero message length
    assisted = df[df['smartwatch_assisted_message_length'] > 0].copy()

    # Calculate changes per 100 characters
    assisted['total_changes_per_100'] = (assisted['smartwatch_assisted_total_changes'] /
                                          assisted['smartwatch_assisted_message_length'] * 100)
    assisted['err_changes_per_100'] = (assisted['smartwatch_assisted_erroneous_changes'] /
                                        assisted['smartwatch_assisted_message_length'] * 100)

    # Melt for Seaborn plotting
    # CORRECTED: Use 'user_id' instead of 'participant_id'
    changes_df = assisted.melt(id_vars=['user_id'],
                               value_vars=['total_changes_per_100', 'err_changes_per_100'],
                               var_name='change_type',
                               value_name='changes_per_100')

    # Clean up change type names
    change_type_map = {
        'total_changes_per_100': 'Total Changes',
        'err_changes_per_100': 'Erroneous Changes'
    }
    changes_df['change_type'] = changes_df['change_type'].map(change_type_map)

    # Define order
    change_type_order = ['Total Changes', 'Erroneous Changes']

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(7, 6)) # Adjusted figure size
    sns.barplot(
        data=changes_df.dropna(subset=['changes_per_100']),
        x='change_type',
        y='changes_per_100',
        order=change_type_order,
        palette="viridis", # Use a consistent or related palette
        capsize=0.1,
        errorbar='sd', # Show standard deviation
        ax=ax
    )

    # --- Customization ---
    ax.set_title('Avg. Changes per 100 Chars (Smartwatch + Intervention)', fontsize=16, pad=20)
    ax.set_xlabel('Type of Change', fontsize=12)
    ax.set_ylabel('Changes per 100 Characters', fontsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    sns.despine() # Remove top and right spines

    # --- Saving ---
    filename = os.path.join(output_dir, 'assisted_changes_barchart.png')
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved plot: {filename}")
    plt.close(fig) # Close the figure to free memory


# --- Main Execution ---
if __name__ == "__main__":
    # Ensure output directory exists
    ensure_dir(OUTPUT_DIR)

    # Load data (Make sure 'session-metrics.csv' is in the same directory or provide the full path)
    try:
        df = pd.read_csv('session-metrics.csv')
        print("Successfully loaded session-metrics.csv")
        print(f"Columns in loaded DataFrame: {df.columns.tolist()}") # Print columns after loading
    except FileNotFoundError:
        print("Error: 'session-metrics.csv' not found. Please place the file in the correct directory.")
        exit() # Exit if the file doesn't exist
    except Exception as e:
        print(f"Error loading or reading CSV: {e}")
        exit()

    # Aggregate Task 3
    if 'task_name' in df.columns:
        df['task_group'] = df['task_name'].replace({'Task 3.1': 'Task 3', 'Task 3.2': 'Task 3'})
    else:
        print("Error: 'task_name' column not found in DataFrame.")
        exit()

    # --- Data Cleaning/Preparation (Example: Ensure numeric types) ---
    # It's good practice to ensure columns you perform calculations on are numeric
    numeric_cols = [
        'phone_time_taken', 'smartwatch_unassisted_time_taken', 'smartwatch_assisted_time_taken',
        'smartwatch_unassisted_non_contiguous_errors_per_100_chars',
        'smartwatch_assisted_non_contiguous_errors_per_100_chars',
        'smartwatch_assisted_message_length', 'smartwatch_assisted_total_changes',
        'smartwatch_assisted_erroneous_changes'
    ]
    missing_cols = [col for col in numeric_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: The following expected numeric columns are missing from the CSV: {missing_cols}")
        exit()

    for col in numeric_cols:
        # Convert to numeric, coercing errors (non-numeric values become NaN)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Optional: Check if entire column became NaN after coercion
        if df[col].isnull().all():
            print(f"Warning: Column '{col}' contains no valid numeric data after conversion.")

    # Generate and save plots
    print("Generating plots...")
    plot_time_taken_boxplots(df, OUTPUT_DIR)
    plot_smartwatch_errors_barchart(df, OUTPUT_DIR)
    plot_assisted_changes_barchart(df, OUTPUT_DIR)

    print(f"All plots generated and saved in '{OUTPUT_DIR}'.")
