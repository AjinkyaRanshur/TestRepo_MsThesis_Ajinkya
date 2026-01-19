"""
Main Interface for Pred-Net Analyzer
FIXED: Added post-processing aggregation option
"""

import os
import sys
from colorama import Fore, Style, init
from utils import clear, banner
from menu_options import (
    main_menu,
    slurm_job_type_menu,
    slurm_training_type_menu,
    slurm_testing_type_menu,
    slurm_recon_entries,
    slurm_classification_entries,
    slurm_select_test_models,
    slurm_get_test_timesteps,
    slurm_get_test_patterns,
    slurm_get_grid_search_params,
    view_registry,
    interactive_mode_placeholder,
    run_post_processing
)

# Initialize colorama
init(autoreset=True)


def handle_slurm_training():
    """Handle SLURM training job submission"""
    training_type = slurm_training_type_menu()
    
    if training_type == "0":
        return
    
    # Get training parameters
    if training_type == "1":
        # Reconstruction training
        base_config = slurm_recon_entries()
    elif training_type == "2":
        # Classification training
        base_config = slurm_classification_entries()
    else:
        print(Fore.RED + "Invalid choice!")
        input("Press ENTER...")
        return
    
    if not base_config:
        return
    
    # Create and submit SLURM script
    from batch_submissions import create_slurm_script, submit_sbatch
    
    script_path, model_names = create_slurm_script(base_config)
    
    print(f"\n{Fore.YELLOW}Review the configuration above.")
    confirm = input("Submit to SLURM? (y/n): ").strip().lower()
    
    if confirm == 'y':
        submitted_info = submit_sbatch(script_path, model_names)
        print(f"\n{Fore.GREEN}✓ Training job submitted!")
        print(f"{Fore.CYAN}Check status with: squeue -u $USER")
        print(f"{Fore.CYAN}View output in: slurm_jobs/")
        print(f"{Fore.CYAN}Aggregate plots will be generated automatically after completion")
    else:
        print(f"\n{Fore.YELLOW}Submission cancelled.")
    
    input("\nPress ENTER to continue...")


def handle_slurm_testing():
    """Handle SLURM testing job submission"""
    testing_type = slurm_testing_type_menu()
    
    if testing_type == "0":
        return
    
    # Get test configuration
    test_config = {}
    
    # Step 1: Select models
    model_names = slurm_select_test_models()
    if not model_names:
        return
    
    test_config["model_names"] = model_names
    
    # Step 2: Get test timesteps
    test_timesteps = slurm_get_test_timesteps()
    test_config["test_timesteps"] = test_timesteps
    
    # Step 3: Test-specific parameters
    if testing_type == "1":
        # Trajectory testing
        test_config["test_type"] = "trajectory"
    
    elif testing_type == "2":
        # Pattern testing
        test_patterns = slurm_get_test_patterns()
        if not test_patterns:
            print(Fore.RED + "No patterns selected!")
            input("Press ENTER...")
            return
        
        test_config["test_type"] = "pattern"
        test_config["test_patterns"] = test_patterns
    
    elif testing_type == "3":
        # Grid search
        grid_params = slurm_get_grid_search_params()
        test_config["test_type"] = "grid_search"
        test_config["grid_params"] = grid_params
    
    else:
        print(Fore.RED + "Invalid choice!")
        input("Press ENTER...")
        return
    
    # Create and submit SLURM test script
    from slurm_testing_submission import create_slurm_test_script, submit_test_job
    
    script_path = create_slurm_test_script(test_config)
    
    print(f"\n{Fore.YELLOW}Review the configuration above.")
    confirm = input("Submit testing job to SLURM? (y/n): ").strip().lower()
    
    if confirm == 'y':
        job_id = submit_test_job(script_path, test_config)
        if job_id:
            print(f"\n{Fore.GREEN}✓ Testing job submitted!")
            print(f"{Fore.CYAN}Check status with: squeue -u $USER")
            print(f"{Fore.CYAN}View output in: slurm_jobs/")
            print(f"{Fore.CYAN}Results will be in: plots/{test_config['test_type']}/")
    else:
        print(f"\n{Fore.YELLOW}Submission cancelled.")
    
    input("\nPress ENTER to continue...")


def main():
    """Main interface loop"""
    while True:
        choice = main_menu()
        
        if choice == "1":
            # SLURM Job Submission
            job_type = slurm_job_type_menu()
            
            if job_type == "1":
                # Training job
                handle_slurm_training()
            elif job_type == "2":
                # Testing job
                handle_slurm_testing()
            elif job_type == "0":
                continue
            else:
                print(Fore.RED + "Invalid option!")
                input("Press ENTER...")
        
        elif choice == "2":
            # View Model Registry
            view_registry()
        
        elif choice == "3":
            # Interactive Mode (placeholder)
            interactive_mode_placeholder()
        
        elif choice == "4":
            # NEW: Generate Aggregate Plots
            run_post_processing()
        
        elif choice == "0":
            # Exit
            clear()
            banner("Goodbye!")
            print(Fore.GREEN + "Thank you for using Pred-Net Analyzer!\n")
            break
        
        else:
            print(Fore.RED + "Invalid option!")
            input("Press ENTER...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Fore.RED}ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
