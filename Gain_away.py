import os
import json
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import linregress
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

def peaks(values, height=10, n_channels=128, mean=None, y_title = "Frequency"):
    # Počet binů – pokud je n_channels pole, vezmeme jeho délku
    if not isinstance(n_channels, int):
        n_bins = len(n_channels)
    else:
        n_bins = n_channels

    # Histogram
    counts, bin_edges = values, np.arange(0, len(values) + 1, 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Hledání vrcholů v histogramu
    peaks_idx, _ = find_peaks(counts, height= mean*height)
    if len(peaks_idx) == 0:
         return None
    # Gaussian funkce
    def gaussian(x, amp, mu, sigma, baseline = mean):
        return baseline + amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    fit_params = []
    for peak in peaks_idx:
        left = max(peak - 5, 0)
        right = min(peak + 6, len(bin_centers))

        x_fit = bin_centers[left:right]
        y_fit = counts[left:right]

        amp_guess = counts[peak]
        mu_guess = bin_centers[peak]
        sigma_guess = (bin_edges[1] - bin_edges[0]) * 3

        try:
            popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=[amp_guess, mu_guess, sigma_guess])
            fit_params.append(popt)
        except RuntimeError:
            continue

     
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0],
            alpha=0.6, label='Histogram')

    x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 1000)
    for i, (amp, mu, sigma) in enumerate(fit_params):
        plt.plot(x_smooth, gaussian(x_smooth, amp, mu, sigma),
                 label=f'Gaussian peak {i + 1}')
    plt.ylim(0, max(counts) * 1.1)
    plt.xlabel('Channel Number')
    plt.ylabel(y_title)
    plt.title('Histogram with Gaussian Peaks')
    plt.legend()
    plt.show()

    return fit_params, bin_centers, counts, bin_edges

def plot_files(variables, folder_path = r"C:/sfg/json", namef=None, datef=None, chips = False, height=1.15):
    """ chips if False, plots all chips in one graph, if True, plots seperate graphs for chips"""
    results = {}
    for var in variables: 
        if var == "gain_away":
            y_title = "Gain (MV/fC)"
            title = f"Gain for {namef} away on {datef}"
            title_ = "Gain away"
        elif var == "gain_under":
            y_title = "Gain (MV/fC)"
            title = f"Gain for {namef} under on {datef}"
            title_ = "Gain under"
        elif var == "innse_away":
            y_title = "Input Noise (ENC)"
            title = f"Input Noice for {namef} away on {datef}"
            title_ = "Input Noice away"
        elif var == "innse_under":
            y_title = "Input Noise (ENC)"
            title = f"Input Noice for {namef} under on {datef}"
            title_ = "Input Noice under"
        elif var == "vt50_away":
            y_title = " Vt50(mV)"
            title = f"Vt50 for {namef} away on {datef}"
            title_ = "Vt50 away"
        elif var == "vt50_under":
            y_title = " Vt50(mV)"
            title = f"Vt50 for {namef} under on {datef}"
            title_ = "Vt50 under"
        
        files = [f for f in os.listdir(folder_path) if f.endswith('.json') and f.startswith(namef) and datef in f]
        values = []
        labels = []
        mask = []

        if chips == False:
            colors = plt.cm.tab10.colors

            fig,axs = plt.subplots(1,2,figsize=(12, 6))
            

            means = []
            errors = []
            labels = []
            counter = 0
            
            for file in files:
                file_path = os.path.join(folder_path, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)

                get = data.get('results', {}).get(var, {})
                if not get:
                    continue
                temp = data.get('properties', {}).get("DCS", {}).get("AMAC_NTCx", "Unknown")
                if temp > 20:
                    mask.append(1)
                else:
                    mask.append(0)
                    
                det_name = data.get('properties', {}).get("det_info", {}).get("name", "Unknown")
                file_date = data.get('date')
                all_channels = []
            

                for chip in get:
                    all_channels.extend(chip)

                n_channels = np.arange(1, len(all_channels) + 1)
                values.extend(all_channels)

                slope, intercept, *_ = linregress(n_channels, all_channels)
                y_fit = slope * n_channels + intercept
                residuals = all_channels - y_fit
                fit_err = np.sqrt(np.sum(residuals**2) / (len(all_channels) - 2))

                means.append(np.mean(all_channels))
                errors.append(fit_err)
                labels.append(file_date)

                peaks(all_channels, height=height, n_channels=n_channels, mean = np.mean(all_channels), y_title=y_title)

                axs[0].plot(n_channels, all_channels, marker='.', linestyle='', markersize=5,
                        label=f"{file_date}",alpha = 0.6, color = colors[counter % len(colors)])
                axs[0].set_xticks(np.arange(0, len(all_channels) + 1, step=128))
                axs[0].plot(n_channels, y_fit,color = colors[counter % len(colors)])
                counter += 1
            

            axs[0].set_xlabel("Channel Number")
            axs[0].set_ylabel(y_title)
            axs[0].set_ylim(min(values)*0.6, math.ceil(max(values) / 100) * 100)
            axs[0].set_xlim(0, max(n_channels) + 1)
            axs[0].set_title(title_)
            axs[0].grid(True)
            axs[0].legend()

            mask = np.array(mask)
            means = np.array(means)
            
            x_pos = np.arange(len(means))
            
            x_pos_w = x_pos[mask == 1]
            means_w = means[mask == 1]
            x_pos_c = x_pos[mask == 0]
            means_c = means[mask == 0]
            if len(x_pos_w) > 0:
                mw, bw, *_ = linregress(x_pos_w, means_w)
                axs[1].plot(x_pos_w, mw*x_pos_w + bw, linestyle='-', label="fit (warm)")
            if len(x_pos_c) > 0:
                mc, bc, *_ = linregress(x_pos_c, means_c)
                axs[1].plot(x_pos_c, mc*x_pos_c + bc, linestyle='-', label="fit (cold)")

            axs[1].errorbar(x_pos, means, yerr=errors, fmt='o', capsize=5)
            axs[1].set_xticks(x_pos, labels, rotation=45)
            axs[1].set_ylabel(f"Mean ± fit error, {y_title}")
            axs[1].set_title(f"Mean values with fit errors, {title_}")
            axs[1].set_ylim(min(means)*0.6, math.ceil(max(means) / 100) * 100)
            axs[1].set_xlim(-1, len(means))
            axs[1].grid(True)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            axs[1].legend()
            fig.suptitle(f"{title}", fontsize=16)
            fig.savefig(f"C:\\sfg\\Code\\Graphs\\{var}_mean_fit_error_{namef}_{datef}.png", dpi=300)
            plt.close(fig)
            results = {
                "component": namef,
                "date": datef,
            }
            results[var] = {
                    "means": list(means),
                    "errors": list(errors)
            }
            
            
            

        else:
            needed_files = []

            for file in files:
                file_path = os.path.join(folder_path, file)

                with open(file_path, 'r') as f:
                    data = json.load(f)

                get = data.get('results', {}).get(var,{})
                if not get:
                    continue

                temp = data.get('properties', {}).get("DCS", {}).get("AMAC_NTCx", "Unknown")
                if temp > 20:
                    mask.append(1)
                else:
                    mask.append(0)

                det_name = data.get('properties', {}).get("det_info", {}).get("name", "Unknown")
                file_date = data.get('date')
                needed_files.append(file)
                
                for chip in get:
                    values.append(chip)

                labels.append(file_date)

            colors = plt.cm.tab10.colors

            
        
            fig1, axs1 = plt.subplots(2, 3, figsize=(15, 8))
            axs1 = axs1.ravel()


            fig2, axs2 = plt.subplots(2, 3, figsize=(18, 8))
            axs2 = axs2.ravel()
            all_means = []
            all_errors = []
            for i in range(6):
                means = []
                errors = []

                for j in range(len(needed_files)):
                    n_channels = list(range(1+i*128, i*128 + len(values[i]) + 1))
                    all_channels = values[i+j*6]

                    slope, intercept, *_ = linregress(n_channels, all_channels)
                    y_fit = slope * np.array(n_channels) + intercept
                    residuals = all_channels - y_fit
                    fit_err = np.sqrt(np.sum(residuals**2) / (len(all_channels) - 2))

                    means.append(np.mean(all_channels))
                    all_means.append(np.mean(all_channels))
                    errors.append(fit_err)
                    all_errors.append(fit_err)
                    peaks(all_channels, height=height, n_channels=n_channels, mean = np.mean(all_channels), y_title=y_title )

                    # --- první graf ---
                    axs1[i].plot(n_channels, all_channels, marker='.', linestyle='', markersize=5, color=colors[j % len(colors)])
                    axs1[i].plot(n_channels, y_fit, color=colors[j % len(colors)])

                axs1[i].set_title(f"Chip {i+1}")
                axs1[i].set_xlabel("Channel Number")
                axs1[i].set_ylabel(y_title)
                axs1[i].set_ylim(min(values[i])*0.6, math.ceil(max(values[i]) / 100) * 100)
                axs1[i].set_xlim(min(n_channels), max(n_channels) + 1)
                axs1[i].grid(True)
                axs1[i].legend()

                # --- druhý graf ---
                mask_arr = np.array(mask)
                means = np.array(means)
                x_pos = np.arange(len(means))

                x_pos_w = x_pos[mask_arr == 1]
                means_w = means[mask_arr == 1]
                x_pos_c = x_pos[mask_arr == 0]
                means_c = means[mask_arr == 0]

                if len(x_pos_w) > 0:
                    mw, bw, *_ = linregress(x_pos_w, means_w)
                    axs2[i].plot(x_pos_w + 1, mw * x_pos_w + bw, linestyle='-', label="fit means (warm)")
                if len(x_pos_c) > 0:
                    mc, bc, *_ = linregress(x_pos_c, means_c)
                    axs2[i].plot(x_pos_c + 1, mc * x_pos_c + bc, linestyle='-', label="fit means (cold)")

                axs2[i].errorbar(x_pos + 1 , means, yerr=errors, fmt='o', capsize=5)
                # axs2[i].set_xticks(x_pos)
                # axs2[i].set_xticklabels(labels, rotation=45)
                axs2[i].set_ylabel(f"{y_title} mean ± fit error")
                axs2[i].set_xlabel("Test Number")
                axs2[i].set_title(f"Chip {i+1}")
                axs2[i].set_ylim(min(means) *0.6, math.ceil(max(means) / 100) * 120)
                axs2[i].set_xlim(0, len(means) + 1)
                axs2[i].grid(True)
                axs2[i].legend()




            fig1.suptitle(f"{title} for all chips", fontsize=16)
            fig2.suptitle(f"Mean values with fit errors, {title} for all chips", fontsize=16)

            fig1.tight_layout(rect=[0, 0, 1, 0.95])
            fig1.subplots_adjust(hspace=0.4)

            fig2.tight_layout(rect=[0, 0, 1, 0.95])
            fig2.subplots_adjust(hspace=0.4)

            fig1.savefig(f"C:\\sfg\\Code\\Graphs\\{var}_single_chips_{namef}_{datef}.png", dpi=300)
            plt.close(fig1)
            fig2.savefig(f"C:\\sfg\\Code\\Graphs\\{var}_mean_fit_error_single_chips{namef}_{datef}.png", dpi=300)
            plt.close(fig2)

            plt.close()
            chunks_all_means = [all_means[i:i + 6] for i in range(0, len(all_means), 6)]
            chunks_errors = [all_errors[i:i + 6] for i in range(0, len(all_errors), 6)]
            results = {
                "component": namef,
                "date": datef,
            }
            if var not in results:
                results[var] = {}

            for i in range(len(chunks_all_means)):
                chip_name = f"Chip_{i+1}"
                results[var][chip_name] = {
                    "means": list(chunks_all_means[i]),
                    "errors": list(chunks_errors[i])
                }
    with open(f"C:\\sfg\\Code\\Graphs\\{namef}_{datef}json", "w") as f:
        json.dump(results, f, indent=4)
    return results
            
    
plot_files(["innse_away"], folder_path=r"C:/sfg/json", namef="SN20USEH40000148_H1", datef="20250702",chips = True, height = 1.05)

