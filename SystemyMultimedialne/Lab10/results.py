from tokenize import Name
import pandas as pd
from sklearn.linear_model import LinearRegression
from main import *

results = pd.read_csv("Lab10/ciasto - szum.csv")

imgs = []
for i in range(10):
    imgs.append("img" + str(i))

edited1 = compress_jpg(img, 90)
edited2 = compress_jpg(img, 80)
edited3 = compress_jpg(img, 70)
edited4 = compress_jpg(img, 60)
edited5 = compress_jpg(img, 50)
edited6 = compress_jpg(img, 40)
edited7 = compress_jpg(img, 30)
edited8 = compress_jpg(img, 25)
edited9 = compress_jpg(img, 15)
edited10 = compress_jpg(img, 5)

# norms = [
#     nmse(noise_SnP(img), img),
#     nmse(noise_gauss(img, 0.3), img),
#     nmse(noise_gauss(img, 0.65), img),
#     nmse(noise_gauss(img, 0.7), img),
#     nmse(noise_gauss(img, 0.9), img),
#     nmse(noise_vals(img, 0.1), img),
#     nmse(noise_vals(img, 0.005), img),
#     nmse(noise_rand(img, 0.9), img),
#     nmse(noise_rand(img, 0.65), img),
#     nmse(noise_rand(img, 0.45), img),
# ]

# norms = [
#     mse(edited1, img),
#     mse(edited2, img),
#     mse(edited3, img),
#     mse(edited4, img),
#     mse(edited5, img),
#     mse(edited6, img),
#     mse(edited7, img),
#     mse(edited8, img),
#     mse(edited9, img),
#     mse(edited10, img),
# ]


norms = [2209.044383, 56.369219, 256.741198, 256.741198, 475.272304, 7.552477, 143.069546, 155.411983, 79.662363, 36.820629]



if __name__ == "__main__":
    
    # user_name = results.iloc[0, 1]
    # user = results.iloc[0, 1:]


    base=pd.DataFrame(data=results).transpose()
    badani = base.iloc[1]
    base = base.drop(base.index[[0, 1]])


    base.index = imgs

    base.assign(Name="norms")
    base.insert(0, "norms", norms)

    col = ["norms"]
    for badany in badani:
        col.append(badany)
    base.columns = col

    base = base.reindex(sorted(base.columns), axis=1)

    mos = []

    for i in range(len(imgs)):
        mos.append(list(base.iloc[i, :-1].values))

    print(mos)

    # symbol = []

    print(base)

    # for _ in range(3):
    #     symbol.append("r*")
    # for _ in range(3):
    #     symbol.append("g^")
    # for _ in range(3):
    #     symbol.append("bv")
 
    for i in range(len(imgs)):
        plt.plot(i, mos[i][0], "r*")
        plt.plot(i, mos[i][1], "r*")
        plt.plot(i, mos[i][2], "r*")
        plt.plot(i, mos[i][3], "g^")
        plt.plot(i, mos[i][4], "g^")
        plt.plot(i, mos[i][5], "g^")
        plt.plot(i, mos[i][6], "bv")
        plt.plot(i, mos[i][7], "bv")
        plt.plot(i, mos[i][8], "bv")
    plt.xticks(np.arange(len(imgs)), imgs)
    plt.show()

    """agregated dla badanego"""

    symbol1 = ["r*", "g^"]
    for i in range(len(imgs)):
        plt.plot(i, np.sum(mos[i][0:3])/3, "r*")
        plt.plot(i, np.sum(mos[i][3:6])/3, "g^")
        plt.plot(i, np.sum(mos[i][6:])/3, "bv")
    plt.xticks(np.arange(len(imgs)), imgs)
    plt.show()

    """agreggated"""
    for i in range(len(imgs)):
        plt.plot(i, np.sum(mos[i][:])/9, "r*")
    plt.xticks(np.arange(len(imgs)), imgs)
    plt.show()

    """mos + miary"""

    base = base.sort_values(by=['norms'])
    print(base)

    # model = LinearRegression()
    # model.fit(base.iloc[:, -1].values, mos[:][:])

    mos = []

    for i in range(len(imgs)):
        mos.append(list(base.iloc[i, :-1].values))

    for i in range(len(imgs)):
        plt.plot(base.iloc[i, -1], mos[i][0], "r*")
        plt.plot(base.iloc[i, -1], mos[i][1], "r*")
        plt.plot(base.iloc[i, -1], mos[i][2], "r*")
        plt.plot(base.iloc[i, -1], mos[i][3], "g^")
        plt.plot(base.iloc[i, -1], mos[i][4], "g^")
        plt.plot(base.iloc[i, -1], mos[i][5], "g^")
        plt.plot(base.iloc[i, -1], mos[i][6], "bv")
        plt.plot(base.iloc[i, -1], mos[i][7], "bv")
        plt.plot(base.iloc[i, -1], mos[i][8], "bv")
    plt.xticks(np.round(base.iloc[:, -1].values))
    plt.show()

    """agregated dla badanego + mos"""

    symbol1 = ["r*", "g^"]
    for i in range(len(imgs)):
        plt.plot(base.iloc[i, -1], np.sum(mos[i][0:3])/3, "r*")
        plt.plot(base.iloc[i, -1], np.sum(mos[i][3:6])/3, "g^")
        plt.plot(base.iloc[i, -1], np.sum(mos[i][6:])/3, "bv")
    plt.xticks(np.round(base.iloc[:, -1].values))
    plt.show()

    """agreggated"""
    for i in range(len(imgs)):
        plt.plot(base.iloc[i, -1], np.sum(mos[i][:])/9, "r*")
    plt.xticks(np.round(base.iloc[:, -1].values))
    plt.show()







