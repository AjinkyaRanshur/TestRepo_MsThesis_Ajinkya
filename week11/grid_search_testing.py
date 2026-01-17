def run_grid_search(trained_pattern_name, model_config, test_timesteps):
    """Run grid search over gamma and beta values."""
    clear()
    banner("Grid Search")

    recon_t = model_config["recon"]
    class_t = model_config["class"]

    model_name = get_model_name(trained_pattern_name, recon_t, class_t)

    GAMMA_VALUES = np.arange(0.13, 0.44, 0.1)
    BETA_VALUES = np.arange(0.13, 0.44, 0.1)

    results_dict = {}
    total = len(GAMMA_VALUES) * len(BETA_VALUES)
    iterations = 1

    print_box("Grid Search Configuration", [
        f"Model: {model_name}",
        f"Gamma range: {GAMMA_VALUES[0]:.2f} - {GAMMA_VALUES[-1]:.2f}",
        f"Beta range: {BETA_VALUES[0]:.2f} - {BETA_VALUES[-1]:.2f}",
        f"Total experiments: {total}"
    ])

    with tqdm(total=total,
              desc="Grid Search",
              unit="config",
              bar_format='{l_bar}{bar:30}{r_bar}') as pbar:

        for gamma in GAMMA_VALUES:
            for beta in BETA_VALUES:

                pbar.set_postfix_str(f"γ={gamma:.2f}, β={beta:.2f}")

                gamma_pattern = [gamma] * 4
                beta_pattern = [beta] * 4

                try:
                    update_config(
                        gamma_pattern,
                        beta_pattern,
                        f"Grid_g{gamma:.2f}_b{beta:.2f}",
                        model_name,
                        test_timesteps,
                        iterations,
                        None,
                        "data/visual_illusion_dataset"
                    )

                    results = run_and_analyze()

                    if results:
                        class_results = {}
                        for cls_name in ["Square", "Random", "All-in", "All-out"]:
                            if cls_name in results:
                                mean_probs = [
                                    np.mean(p) * 100
                                    for p in results[cls_name]["predictions"]
                                ]
                                class_results[cls_name] = max(mean_probs)

                        results_dict[(gamma, beta)] = class_results

                except Exception as e:
                    print(f"\n{Fore.RED}Error at gamma={gamma:.2f}, beta={beta:.2f}: {e}")
                    results_dict[(gamma, beta)] = {}

                pbar.update(1)

    # Build illusion matrix
    illusion_matrix = np.zeros((len(BETA_VALUES), len(GAMMA_VALUES)))

    for i, beta in enumerate(BETA_VALUES):
        for j, gamma in enumerate(GAMMA_VALUES):

            res = results_dict.get((gamma, beta), {})
            max_allin = res.get("All-in", 0)
            max_allout = res.get("All-out", 0)
            max_random = res.get("Random", 0)

            denom = (max_allout + max_random) / 2
            illusion_index = max_allin / denom if denom > 0 else 0

            illusion_matrix[i, j] = illusion_index

    print("\n")

    plot_grid_heatmap(
        GAMMA_VALUES,
        BETA_VALUES,
        illusion_matrix,
        model_name,
        recon_t,
        class_t
    )

    return results_dict, illusion_matrix
