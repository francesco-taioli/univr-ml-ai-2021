import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure


def plot_MeanIoU_val():
    figure(num=None, figsize=(50, 50), dpi=20, facecolor='w', edgecolor='k')
    font = {'size': '40'}
    fontT = {'weight': 'bold',
             'size': '50'}
    barWidth = 0.6

    MeanIoU = {"LinkNet_288_CL_RMS_Jloss": 0.718,
               "LinkNet_288_WU_RMS_Jloss": 0.6885,
               "PSPNet_288_WU_RMS_Jloss": 0.6909,
               "LinkNet_288_CL_SGD_Jloss": 0.4336,
               "LinkNet_288_CL_RMS_CEloss": 0.5674,
               "LinkNet_288_CL_RMS_PWloss": 0.5602,
               "SegNet_288_WU_RMS_Jloss": 0.578,
               "UNet_288_WU_RMS_Jloss": 0.5987,
               "UNet_288_CL_RMS_JLoss": 0.6285,
               "PSPNet_288_CL_RMS_Jloss": 0.6624,
               "SegNet_288_CL_RMS_Jloss": 0.6795,
               "LinkNet_288_CL_Adam_Jloss": 0.6696,
               "LinkNet_288_ROP_RMS_Jloss": 0.6837,
               "FCN8_288_WU_RMS_Jloss": 0.2739
               }
    sorted_values = sorted(MeanIoU.values())  # Sort the values
    sorted_dict = {}

    for i in sorted_values:
        for k in MeanIoU.keys():
            if MeanIoU[k] == i:
                sorted_dict[k] = MeanIoU[k]
                break
    MeanIoU = sorted_dict
    values = [*MeanIoU.values()]

    positions = [i for i in range(1, len(MeanIoU) + 1)]
    plt.bar(positions, [j for j in values], width=barWidth, color="orange")
    # plt.bar(1, MeanIoU["HSV_HOG_FC"]+0.5, error_kw=dict(lw=5, capsize=5, capthick=3),
    #        width = barWidth+0.1, color = "red", label="Extraction of features (500 points) + inference")
    plt.xticks([r + barWidth + 0.4 for r in range(len(MeanIoU))], MeanIoU.keys(), rotation=45, font=fontT, ha='right')

    # Create labels
    for i in range(0, len(MeanIoU)):
        plt.text(x=i + 0.6, y=[j for j in values][i] + 0.005, s=str(float([j for j in values][i])), font=font)

    plt.subplots_adjust(bottom=0.2, top=0.98)
    plt.xlabel('Models', fontT)
    plt.ylabel('MeanIoU', fontT)
    plt.title('Best validation meanIoU', fontT)
    plt.yticks(np.arange(0, 1, 0.25), [0, 0.25, 0.5, 0.75], font=fontT)

    plt.show()


if __name__ == "__main__":
    plot_MeanIoU_val()
