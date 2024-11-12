import numpy as np 
from src.style_time import style_time
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm 



def get_path(full_path:str):
    return "/".join(full_path.split("/")[:-1])
    

def get_name(full_path:str):
    filename= "".join(full_path.split("/")[-1])
    return filename.split(".")[0]


def plot_loss_curves(content_losses, style_losses, total_losses, save_to):

    c_maxes = np.max(content_losses, axis=0)
    c_mins = np.min(content_losses, axis=0)
    c_means = np.mean(content_losses, axis=0)

    s_maxes = np.max(style_losses, axis=0)
    s_mins = np.min(style_losses, axis=0)
    s_means = np.mean(style_losses, axis=0)

    t_maxes = np.max(total_losses, axis=0)
    t_mins = np.min(total_losses, axis=0)
    t_means = np.mean(total_losses, axis=0)

    t = np.arange(c_maxes.shape[0])

    plt.figure(figsize=(18, 5))

    ax = plt.subplot(311)
    ax.set_title("Content Loses Variations")

    plt.plot(c_mins)
    plt.plot(c_maxes)

    ax.fill_between(t, c_mins, c_maxes, alpha=0.5)
    plt.plot(c_means)

    ax.grid(True)

    ax = plt.subplot(312)
    ax.set_title("Style Loses Variations")

    plt.plot(s_mins)
    plt.plot(s_maxes)

    ax.fill_between(t, s_mins, s_maxes, alpha=0.5)

    plt.plot(s_means)
    ax.grid(True)

    ax = plt.subplot(313)
    ax.set_title("Total Loses Variations")

    plt.plot(t_mins)
    plt.plot(t_maxes)

    ax.fill_between(t, t_mins, t_maxes, alpha=0.5)


    plt.plot(t_means)
    ax.grid(True)
    
    plt.savefig(save_to)

    plt.show()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("content_dset_path", type=str)
    parser.add_argument("style_dset_path", type=str)
    args = parser.parse_args()

    path = get_path(args.content_dset_path)
    filename = get_name(args.content_dset_path)
    

    perturbed_content_train = np.load(args.content_dset_path)
    style_train = np.load(args.style_dset_path)
    transfered_style = []
    c_losses, s_losses, total_losses = [], [], []
    
    

    for (c_seq, s_seq) in tqdm(zip(perturbed_content_train, style_train),total=perturbed_content_train.shape[0]):
        results, cl, sl, tl = style_time(c_seq, s_seq, 1000, verbose=0, alpha=1000., beta=50., learning_rate=0.01)
        transfered_style.append(results)
        
        c_losses.append(cl)
        s_losses.append(sl)
        total_losses.append(tl)
        
    transfered_style = np.array(transfered_style)
    
    c_losses = np.array(c_losses)
    s_losses = np.array(s_losses)
    total_losses = np.array(total_losses)
    
    plot_loss_curves(c_losses, s_losses, total_losses, f"{path}/losses.png")
    
    np.save(f"{path}/{filename}_transfered.npy", transfered_style)

if __name__ == "__main__":
    main()
    
    
    
#  python paper_style_transfer.py "data/google_stocks/preprocessed/normal_content_train.npy" "data/google_stocks/preprocessed/style_train.npy"
#  python paper_style_transfer.py "data/google_stocks/preprocessed/normal_content_test.npy" "data/google_stocks/preprocessed/style_test.npy"

#  python paper_style_transfer.py "data/google_stocks/preprocessed/perturbed_content_train.npy" "data/google_stocks/preprocessed/style_train.npy"
#  python paper_style_transfer.py "data/google_stocks/preprocessed/perturbed_content_test.npy" "data/google_stocks/preprocessed/style_test.npy"

########

#  python paper_style_transfer.py "data/energy/preprocessed/in_sample_train.npy" "data/energy/preprocessed/style_train.npy"
#  python paper_style_transfer.py "data/energy/preprocessed/in_sample_test.npy" "data/energy/preprocessed/style_test.npy"

#  python paper_style_transfer.py "data/energy/preprocessed/perturbed_train.npy" "data/energy/preprocessed/style_train.npy"
#  python paper_style_transfer.py "data/energy/preprocessed/perturbed_test.npy" "data/energy/preprocessed/style_test.npy"
