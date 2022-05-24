    fig, axs = plt.subplots(4, 1 , sharey=True)
    fig.set_size_inches(9,13)
    axs[0].imshow(data)
    axs[1].imshow(Y,cmap=plt.cm.gray)
    axs[2].imshow(Cr,cmap=plt.cm.gray)
    axs[3].imshow(Cb,cmap=plt.cm.gray)
    plt.show()

    print(Cb)


    axs[0,0].imshow(data)
    axs[1,0].imshow(Y, cmap=plt.cm.gray)
    axs[2,0].imshow(Cr, cmap=plt.cm.gray)
    axs[3,0].imshow(Cb, cmap=plt.cm.gray)

    sampling = "4:2:2"
    one = False
    Y2 = test(Y, sampling, QY)
    Cr2 = test(Cr, QC)
    Cb2 = test(Cb, QC)

    axs[0,1].imshow(np.dstack([Y2,Cr2,Cb2]).astype(np.uint8))
    axs[1,1].imshow(Y2, cmap=plt.cm.gray)
    axs[2,1].imshow(Cr2, cmap=plt.cm.gray)
    axs[3,1].imshow(Cb2, cmap=plt.cm.gray)

    plt.show()