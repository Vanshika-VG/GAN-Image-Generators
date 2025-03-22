def main():
    print("Loading Module.....")

    import torch
    from torch import nn
    from torch.utils.data import DataLoader,random_split
    from torchvision import datasets
    from torchvision import transforms
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    from tqdm.auto import tqdm
    import numpy as np

    import Networks as models
    from utils import gradient_penalty

    print("Modules Loaded.....")

    DEVICE = "cuda"

    BATCH_SIZE = 128
    LEARNING_RATE_CRITIC = 1e-5
    LEARNING_RATE_GENER = 1e-5
    NUM_OF_EPOCHS = 3000
    NUM_OF_CRITIC_STEP = 5
    NUM_OF_GENER_STEP = 1
    FEATURES_CRITIC = 64
    FEATURES_GEN = 64
    LAMBDA_GP = 10   
    LOAD_MODELS = True
    IMAGE_HEIGHT,IMAGE_WIDTH,COLOR_CHANNELS = 64,64,3
    LATENT_DIM = 100

    SHOW_OUTPUT = 1
    SAVE_FIG = 10


    data_transform = transforms.Compose(
        [transforms.Resize((IMAGE_HEIGHT,IMAGE_WIDTH)),
         transforms.ToTensor(),
         transforms.Normalize([0.5 for _ in range(COLOR_CHANNELS)],[0.5 for _ in range(COLOR_CHANNELS)])
        ])

    train_data = datasets.ImageFolder("Training",transform=data_transform)

    data_loader = DataLoader(dataset=train_data,shuffle=True,batch_size=BATCH_SIZE,pin_memory=True)

    gen = models.Generator(LATENT_DIM, COLOR_CHANNELS,FEATURES_GEN).to(DEVICE)
    critic = models.Discriminator(COLOR_CHANNELS,FEATURES_CRITIC).to(DEVICE)

    if(LOAD_MODELS):
        gen.load_state_dict(torch.load("TrainingModels/generator.pth",weights_only=True))
        critic.load_state_dict(torch.load("TrainingModels/discriminator.pth",weights_only=True))
    else:
        models.initialize_weights(gen)
        models.initialize_weights(critic)


    opt_gen = torch.optim.Adam(gen.parameters(),lr=LEARNING_RATE_GENER,betas=(0.0,0.9))
    opt_critic = torch.optim.Adam(critic.parameters(),lr=LEARNING_RATE_CRITIC,betas=(0.0,0.9))

    fixed_noise = torch.randn(32,LATENT_DIM,1,1).to(DEVICE)

    gen.train()
    critic.train()

    critic_losses = []
    gen_losses = []

    for epoch in range(NUM_OF_EPOCHS):
        for batch, (real,_) in enumerate(tqdm(data_loader)):
            real = real.to(DEVICE)

            #Critic Training
            for _ in range(NUM_OF_CRITIC_STEP):
                noise = torch.randn((real.shape[0],LATENT_DIM,1,1)).to(DEVICE)
                fake = gen(noise)
                critic_real = critic(real).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                gp = gradient_penalty(critic,real,fake,DEVICE)
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp

                critic.zero_grad()
                loss_critic.backward(retain_graph = True)
                opt_critic.step()

            #Generator Training
            for _ in range(NUM_OF_GENER_STEP):
                output = critic(fake).reshape(-1)
                loss_gen = -torch.mean(output)
                gen.zero_grad()
                loss_gen.backward()
                opt_gen.step()

        print(f"Epoch: [{epoch}/{NUM_OF_EPOCHS}] Critic Loss: {loss_critic:.8f} | Gen loss: {loss_gen:.8f}")
        critic_losses.append(loss_critic.detach().cpu())
        gen_losses.append(loss_gen.detach().cpu())

        torch.save(gen.state_dict(),"TrainingModels/generator.pth")
        torch.save(critic.state_dict(),"TrainingModels/discriminator.pth")
        print("Models saved.")

        #Show model output
        if((epoch + 1) % SHOW_OUTPUT == 0):        
            gen.eval()

            with torch.inference_mode():
                fake = (gen(fixed_noise)).to("cpu")

                img_grid_real = make_grid(
                    real[:32],normalize=True
                )
                img_grid_fake = make_grid(
                    fake[:32] , normalize= True
                )
                plt.imshow(np.transpose(img_grid_fake, (1, 2, 0)))
                plt.axis("off")
                plt.show()
                plt.plot(gen_losses,c= 'r')
                plt.plot(critic_losses,c = 'b')
                plt.show()

            gen.train()

        #Save figure
        if((epoch + 1) % SAVE_FIG == 0):        
            plt.plot(gen_losses,c= 'r')
            plt.plot(critic_losses,c = 'b')
            plt.savefig("Figures/Epoch_" + str(epoch) + ".png", bbox_inches='tight')
                
                    
            

if __name__ == '__main__':
    main()