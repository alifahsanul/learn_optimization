"""
Simulated Annealing Algorithm Implementation
==========================================

Simulated Annealing is a probabilistic optimization algorithm inspired by the
physical process of annealing in metallurgy. It's used to find approximate
solutions to optimization problems.

Key Concepts:
- Temperature: Controls the probability of accepting worse solutions
- Cooling Schedule: How temperature decreases over time
- Acceptance Probability: P(accept) = exp(-ΔE/T) where ΔE is the energy difference
"""

import random
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


class SimulatedAnnealing:
    def __init__(self, initial_temp=1000, cooling_rate=0.95, min_temp=1):
        """
        Initialize the simulated annealing algorithm
        
        Args:
            initial_temp: Starting temperature (higher = more random moves)
            cooling_rate: Rate at which temperature decreases (0.95 = 5% reduction)
            min_temp: Minimum temperature (stopping condition)
        """
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.temperature = initial_temp
        
    def objective_function(self, x):
        """
        Example objective function to minimize
        This is a multimodal function with multiple local minima
        """
        return (x - 2)**2 * (x + 3)**2 + 10 * math.sin(x) + 1
    
    def neighbor_function(self, x, step_size=1.0):
        """
        Generate a neighboring solution
        """
        return x + random.uniform(-step_size, step_size)
    
    def acceptance_probability(self, current_energy, new_energy):
        """
        Calculate probability of accepting a worse solution
        """
        if new_energy < current_energy:
            return 1.0  # Always accept better solutions
        return math.exp(-(new_energy - current_energy) / self.temperature)
    
    def solve(self, initial_solution, max_iterations=1000, save_history=True):
        """
        Main simulated annealing algorithm
        
        Args:
            initial_solution: Starting point for optimization
            max_iterations: Maximum number of iterations
            save_history: Whether to save detailed history for animation
            
        Returns:
            best_solution: Best solution found
            history: List of solutions and energies for plotting
        """
        current_solution = initial_solution
        current_energy = self.objective_function(current_solution)
        
        best_solution = current_solution
        best_energy = current_energy
        
        history = []
        iteration = 0
        
        print("Starting Simulated Annealing...")
        print(f"Initial solution: {current_solution:.4f}, Energy: {current_energy:.4f}")
        print(f"Temperature: {self.temperature:.2f}")
        print("-" * 50)
        
        while self.temperature > self.min_temp and iteration < max_iterations:
            # Generate neighboring solution
            new_solution = self.neighbor_function(current_solution)
            new_energy = self.objective_function(new_solution)
            
            # Calculate acceptance probability
            acceptance_prob = self.acceptance_probability(current_energy, new_energy)
            
            # Accept or reject the new solution
            accepted = random.random() < acceptance_prob
            if accepted:
                current_solution = new_solution
                current_energy = new_energy
                
                # Update best solution if necessary
                if current_energy < best_energy:
                    best_solution = current_solution
                    best_energy = current_energy
                    print(f"Iteration {iteration}: New best! Solution: {best_solution:.4f}, Energy: {best_energy:.4f}")
            
            # Record history with detailed info for animation
            if save_history:
                history.append({
                    'iteration': iteration,
                    'current_solution': current_solution,
                    'current_energy': current_energy,
                    'new_solution': new_solution,
                    'new_energy': new_energy,
                    'best_solution': best_solution,
                    'best_energy': best_energy,
                    'temperature': self.temperature,
                    'acceptance_prob': acceptance_prob,
                    'accepted': accepted
                })
            
            # Cool down
            self.temperature *= self.cooling_rate
            iteration += 1
        
        print("-" * 50)
        print(f"Optimization complete after {iteration} iterations")
        print(f"Best solution: {best_solution:.4f}")
        print(f"Best energy: {best_energy:.4f}")
        print(f"Final temperature: {self.temperature:.4f}")
        
        return best_solution, history


def plot_results(history, objective_func):
    """
    Plot the optimization progress and objective function
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Extract data (handle both old and new history formats)
    iterations = [h['iteration'] for h in history]
    
    # Check if it's the new format with detailed history
    if 'current_energy' in history[0]:
        energies = [h['current_energy'] for h in history]
        solutions = [h['current_solution'] for h in history]
    else:
        # Old format
        energies = [h['energy'] for h in history]
        solutions = [h['solution'] for h in history]
    
    temperatures = [h['temperature'] for h in history]
    
    # Plot 1: Energy over iterations (with log scale, handling negative values)
    # Transform: sign(value) * log(|value|) for log scale with negatives
    log_energies = []
    for e in energies:
        if e == 0:
            log_energies.append(0)  # Handle zero case
        else:
            log_energies.append(math.copysign(math.log(abs(e)), e))
    
    ax1.plot(iterations, log_energies, 'b-', alpha=0.7)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('sign(Energy) * log(|Energy|)')
    ax1.set_title('Energy vs Iterations (Log Scale with Negatives)')
    ax1.grid(True)
    
    # Plot 2: Temperature over iterations (log scale)
    ax2.plot(iterations, temperatures, 'r-')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Temperature')
    ax2.set_title('Temperature Schedule')
    ax2.set_yscale('log')  # Temperature always decreases, so log scale makes sense
    ax2.grid(True)
    
    # Plot 3: Objective function with search path
    x_range = np.linspace(-5, 5, 1000)
    y_range = [objective_func(x) for x in x_range]
    
    ax3.plot(x_range, y_range, 'g-', label='Objective Function', linewidth=2)
    ax3.scatter(solutions, energies, c='red', alpha=0.6, s=20, label='Search Path')
    ax3.set_xlabel('Solution')
    ax3.set_ylabel('Energy')
    ax3.set_title('Search Path on Objective Function')
    ax3.legend()
    ax3.grid(True)
    
    # Apply log scale for objective function, handling negative values
    # Transform: sign(value) * log(|value|) for log scale with negatives
    log_y_range = []
    for y in y_range:
        if y == 0:
            log_y_range.append(0)  # Handle zero case
        else:
            log_y_range.append(math.copysign(math.log(abs(y)), y))
    
    log_energies_obj = []
    for e in energies:
        if e == 0:
            log_energies_obj.append(0)  # Handle zero case
        else:
            log_energies_obj.append(math.copysign(math.log(abs(e)), e))
    
    ax3.clear()
    ax3.plot(x_range, log_y_range, 'g-', label='Objective Function', linewidth=2)
    ax3.scatter(solutions, log_energies_obj, c='red', alpha=0.6, s=20, label='Search Path')
    ax3.set_xlabel('Solution')
    ax3.set_ylabel('sign(Energy) * log(|Energy|)')
    ax3.set_title('Search Path on Objective Function (Log Scale with Negatives)')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()


def animate_simulated_annealing(history, objective_func, save_gif=False):
    """
    Create an animated visualization of simulated annealing optimization
    
    Args:
        history: Detailed history from solve() method
        objective_func: The objective function to plot
        save_gif: Whether to save as GIF file
    """
    # Set up the figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Define the range for plotting
    x_range = np.linspace(-5, 5, 1000)
    y_range = [objective_func(x) for x in x_range]
    
    # Plot the objective function (static background)
    ax1.plot(x_range, y_range, 'g-', linewidth=2, alpha=0.7, label='Objective Function')
    ax1.set_xlabel('Solution')
    ax1.set_ylabel('Energy')
    ax1.set_title('Simulated Annealing Animation - Search Process')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Initialize empty line for energy history
    energy_line, = ax2.plot([], [], 'b-', alpha=0.7, label='Energy')
    temp_line, = ax2.plot([], [], 'r-', alpha=0.7, label='Temperature')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Value')
    ax2.set_title('Energy and Temperature over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Initialize scatter points for current and best solutions
    current_point = ax1.scatter([], [], c='red', s=100, marker='o', 
                               label='Current Solution', zorder=5)
    new_point = ax1.scatter([], [], c='orange', s=80, marker='s', 
                           label='Proposed Solution', zorder=5)
    best_point = ax1.scatter([], [], c='blue', s=120, marker='*', 
                            label='Best Solution', zorder=6)
    
    # Text for displaying information
    info_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Lists to store animation data
    iterations = []
    energies = []
    temperatures = []
    
    def animate(frame):
        if frame >= len(history):
            return current_point, new_point, best_point, energy_line, temp_line, info_text
        
        h = history[frame]
        
        # Update solution points
        current_point.set_offsets([[h['current_solution'], h['current_energy']]])
        new_point.set_offsets([[h['new_solution'], h['new_energy']]])
        best_point.set_offsets([[h['best_solution'], h['best_energy']]])
        
        # Update energy and temperature history
        iterations.append(h['iteration'])
        energies.append(h['current_energy'])
        temperatures.append(h['temperature'])
        
        energy_line.set_data(iterations, energies)
        temp_line.set_data(iterations, temperatures)
        
        # Auto-scale the plots
        if len(iterations) > 1:
            ax2.set_xlim(min(iterations), max(iterations))
            if min(energies) > 0:
                ax2.set_ylim(min(min(energies), min(temperatures)), 
                           max(max(energies), max(temperatures)))
        
        # Update information text
        status = "ACCEPTED" if h['accepted'] else "REJECTED"
        info_text.set_text(
            f'Iteration: {h["iteration"]}\n'
            f'Temperature: {h["temperature"]:.2f}\n'
            f'Current: {h["current_solution"]:.3f} (E={h["current_energy"]:.3f})\n'
            f'Proposed: {h["new_solution"]:.3f} (E={h["new_energy"]:.3f})\n'
            f'Best: {h["best_solution"]:.3f} (E={h["best_energy"]:.3f})\n'
            f'Accept Prob: {h["acceptance_prob"]:.3f}\n'
            f'Status: {status}'
        )
        
        return current_point, new_point, best_point, energy_line, temp_line, info_text
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(history), 
                                 interval=100, blit=False, repeat=True)
    
    if save_gif:
        print("Saving animation as GIF... (this may take a moment)")
        anim.save('simulated_annealing_animation.gif', writer='pillow', fps=10)
        print("Animation saved as 'simulated_annealing_animation.gif'")
    
    plt.tight_layout()
    plt.show()
    
    return anim


def main():
    """
    Main function to demonstrate simulated annealing
    """
    print("=== Simulated Annealing Demo ===\n")
    
    # Create optimizer instance
    sa = SimulatedAnnealing(initial_temp=100, cooling_rate=0.95, min_temp=0.1)
    
    # Run optimization with detailed history for animation
    best_solution, history = sa.solve(initial_solution=0.0, max_iterations=100)
    
    # Show static plots first
    plot_results(history, sa.objective_function)
    
    # Show animated version
    print("\nStarting animation...")
    animate_simulated_annealing(history, sa.objective_function, save_gif=False)
    
    # Demonstrate different cooling rates
    print("\n=== Comparing Different Cooling Rates ===")
    cooling_rates = [0.90, 0.95, 0.98, 0.99]
    
    for rate in cooling_rates:
        sa_test = SimulatedAnnealing(initial_temp=100, cooling_rate=rate, min_temp=0.1)
        solution, _ = sa_test.solve(initial_solution=0.0, max_iterations=100, save_history=False)
        energy = sa_test.objective_function(solution)
        print(f"Cooling rate {rate}: Solution = {solution:.4f}, Energy = {energy:.4f}")


def animation_demo():
    """
    Demo specifically for animation with longer run
    """
    print("=== Simulated Annealing Animation Demo ===\n")
    
    # Create optimizer instance
    sa = SimulatedAnnealing(initial_temp=100, cooling_rate=0.95, min_temp=0.1)
    
    # Run optimization with detailed history for animation
    best_solution, history = sa.solve(initial_solution=0.0, max_iterations=200)
    
    # Show animated version and save as GIF
    print("Creating animation...")
    animate_simulated_annealing(history, sa.objective_function, save_gif=True)


if __name__ == "__main__":
    main()
