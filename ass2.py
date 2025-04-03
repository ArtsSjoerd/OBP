import numpy as np
import streamlit as st

def create_transition_matrix(n, k, failure_rate, repair_rate, repairman, warm_standby):
    """
    Creates a matrix consisting of transition probabilities that correspond to the input values
    """
    num_states = n + 1 # States from 0 to n
    
    transition_matrix = np.zeros((num_states, num_states))
    
    for i in range(num_states):
        total_rate = 0
        
        # Transition of a failure, from state i to state i - 1
        if i > 0:
            if warm_standby:
                rate_fail = i * failure_rate # All components can fail at any time
            else:
                rate_fail = k * failure_rate if i >= k else 0 # Only k active components can fail

            transition_matrix[i, i - 1] = rate_fail
            total_rate += rate_fail
            
        # Transition of a repair, from state i to state i + 1
        if i < n:
            rate_repair = min(repairman, n - i) * repair_rate
            transition_matrix[i, i + 1] = rate_repair
        
    # Fill in the diagonal elements
    for i in range(num_states):
        transition_matrix[i, i] = -np.sum(transition_matrix[i, :])
    
    return transition_matrix

def matrix_vector_mult(vector, matrix):
    """
    Performs vector-matrix multiplication manually.
    """
    result = [0.0] * len(vector)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            result[i] += vector[j] * matrix[j][i]  # Matrix [j][i] because input vector pi is a row vector
    return result

def compute_stationary_distribution(Q, tolerance=1e-6, max_iterations=10000, delta_t=0.01):
    """
    Computes the stationary distribution using an iterative Euler method.
    """
    n = len(Q)
    pi = [1.0 / n] * n  # Uniform initial distribution

    for _ in range(max_iterations):
        Qpi = matrix_vector_mult(pi, Q)  # Compute πQ
        pi_new = [pi[i] + delta_t * Qpi[i] for i in range(n)]  # Euler step

        # Normalize (to ensure probabilities sum to 1)
        total = sum(pi_new)
        if total > 0:
            pi_new = [x / total for x in pi_new]

        # Check for convergence
        if sum(abs(a - b) for a, b in zip(pi_new, pi)) < tolerance:
            return pi_new

        pi = pi_new

    return pi

def compute_uptime_probability(pi, k):
    """
    Computes the probability that the system is operational, 
    given a stationary distribution and the number of working components that are required for the system to function
    """
    return sum(pi[k:])

def compute_downtime_cost(pi, k, cost_downtime):
    """
    Computes the costs that are related to the total downtime of the system
    """
    downtime_cost = sum(pi[i] * cost_downtime for i in range(k))
    
    return downtime_cost

def total_cost(n, k, failure_rate, repair_rate, repairman, warm_standby, cost_component, cost_repairman, cost_downtime):
    """
    Computes the total cost of a system for a given input combination
    """
    
    # Create transition matrix
    transition_matrix = create_transition_matrix(n, k, failure_rate, repair_rate, repairman, warm_standby)
    
    # Compute stationary distribution
    pi = compute_stationary_distribution(transition_matrix)
    
    # Calculate downtime cost
    downtime_cost = compute_downtime_cost(pi, k, cost_downtime)
    
    # Calculate component cost
    component_cost = cost_component * n
    
    # Calculate repairman cost
    repairman_cost = cost_repairman * repairman
    
    # Total cost
    return component_cost + repairman_cost + downtime_cost

def find_optimal_parameters(failure_rate, repair_rate, k, warm_standby, cost_component, cost_repairman, cost_downtime, n_range, repairman_range):
    """
    This function find the optimal number of components and the number of repairman to minimze the total costs,
    given the number of working components that are required for the system to function
    """
    
    optimal_cost = float('inf')
    optimal_n = 0
    optimal_repairman = 0
    
    # Brute-force search for the optimal n and repairman values
    for n in n_range:
        for repairman in repairman_range:
            cost = total_cost(n, k, failure_rate, repair_rate, repairman, warm_standby, cost_component, cost_repairman, cost_downtime)
            if cost < optimal_cost:
                optimal_cost = cost
                optimal_n = n
                optimal_repairman = repairman
    
    return optimal_n, optimal_repairman, optimal_cost

def main():
    """
    This method defines the interface provided by Streamlit
    """
    
    # Initialize screen state if not already initialized
    if "screen" not in st.session_state:
        st.session_state.screen = "home"  # Default screen is the home screen
        
    st.title("Analysis of a k-out-of-n maintenance system")

    # Home screen
    if st.session_state.screen == "home":
        option = st.radio("Select an option:", ("Calculate probability that the system is up", "Find optimal number of components and repairman given k"))

        if option == "Calculate probability that the system is up":
            if st.button("Go to the calculation page"):
                st.session_state.screen = "calculate"  # Switch to calculation screen
                st.rerun()

        elif option == "Find optimal number of components and repairman given k":
            if st.button("Go to the optimization page"):
                st.session_state.screen = "optimize"  # Switch to optimization screen
                st.rerun()

    # Calculation screen
    elif st.session_state.screen == "calculate":
        st.subheader("Calculate the probability the system is up")

        n = st.number_input("Number of components (n)", min_value=0, value=1)
        k = st.number_input("Number of components needed for system to function (k)", min_value=0, max_value=n, value=1)
        failure_rate = st.number_input("Failure rate (λ)", min_value=0.0, value=0.0)
        repair_rate = st.number_input("Repair rate (μ)", min_value=0.0, value=0.0)
        repairmen = st.number_input("Number of repairmen (s)", min_value=1, value=1)
        warm_standby = st.selectbox("Warm standby? (Yes/No)", ("Yes", "No"))

        warm_standby = warm_standby == "Yes"

        if st.button("Calculate"):
            # Compute uptime probability
            P = create_transition_matrix(n, k, failure_rate, repair_rate, repairmen, warm_standby)
            pi = compute_stationary_distribution(P)
            uptime = compute_uptime_probability(pi, k)
            
            st.subheader("Fraction of time the system is up")
            st.write(uptime)
            
            # Round the uptime to three decimal places
            uptime = round(uptime, 3)

        if st.button("Back to home"):
            st.session_state.screen = "home"
            st.rerun()

    # Optimization screen
    elif st.session_state.screen == "optimize":
        st.subheader("Find Optimal Parameters")

        k = st.number_input("Number of components needed for system to function (k)", min_value=0, value=1)
        failure_rate = st.number_input("Failure rate (λ)", min_value=0.0, value=0.0)
        repair_rate = st.number_input("Repair rate (μ)", min_value=0.0, value=0.0)
        warm_standby = st.selectbox("Warm standby? (Yes/No)", ("Yes", "No"))
        cost_component = st.number_input("Cost per component", min_value=0, value = 1)
        cost_repairman = st.number_input("Cost per repairman", min_value=0, value = 1)
        cost_downtime = st.number_input("Cost of downtime", min_value=0, value = 1)

        warm_standby = warm_standby == "Yes"

        if st.button("Find Optimal Parameters"):
            # Compute optimal parameters
            max_components = 5
            max_repairmen = 5

            n_range = range(k, k + max_components + 1)  # Range of possible component counts
            repairman_range = range(1, max_repairmen + 1)  # Range of possible repairmen counts

            optimal_n, optimal_repairman, optimal_cost = find_optimal_parameters(failure_rate, repair_rate, 
                                                                                 k, warm_standby, cost_component, cost_repairman, cost_downtime, n_range, repairman_range)

            old_n = 0
            old_repairman = 0

            while optimal_n == (k +  max_components) or optimal_repairman == max_repairmen:
                if optimal_n == old_n and optimal_repairman == old_repairman:
                    break

                old_n = optimal_n
                old_repairman = optimal_repairman

                max_components += 5
                max_repairmen += 5
                optimal_n, optimal_repairman, optimal_cost = find_optimal_parameters(failure_rate, repair_rate, k, 
                                                                                     warm_standby, cost_component, cost_repairman, cost_downtime, 
                                                                                     range(max(k, optimal_n - 5), k + max_components + 1), 
                                                                                     range(max(1, optimal_repairman - 5), max_repairmen + 1))
            
            optimal_cost = round(optimal_cost, 2)
            
            st.subheader("Optimal Parameters")
            st.write(f"Number of components (n): {optimal_n}")
            st.write(f"Number of repairmen (s): {optimal_repairman}")
            st.write(f"Costs: {optimal_cost}")

        if st.button("Back to Home"):
            st.session_state.screen = "home"
            st.rerun()

if __name__ == "__main__":
    main()