def main():
    print("Loading Modules.....")

    import torch
    from utils import CGANDataset
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torch import nn
    from torch import optim
    import NeuralNetworks as models
    from tqdm.auto import tqdm
    from torchvision.utils import make_grid


    print("Modules Loaded.....")

    DEVICE = "cuda"

    BATCH_SIZE = 20
    LEARNING_RATE = 0.0001
    LAMBDA_CYCLE = 6
    LAMBDA_IDENTITY = 0

    LOAD_MODELS = True
    NUM_OF_EPOCHS = 1200
    SHOW = 10

    TRANSFORM = transforms.Compose(
        [
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ],

    )


    disc_Z = models.Discriminator(in_channels=3).to(DEVICE)
    disc_H = models.Discriminator(in_channels=3).to(DEVICE)
    gen_Z = models.Generator(img_channels=3,num_residuals=9).to(DEVICE)
    gen_H = models.Generator(img_channels=3,num_residuals=9).to(DEVICE)

    gen_Z.train()
    gen_H.train()
    disc_Z.train()
    disc_H.train()

    if LOAD_MODELS:
        disc_Z.load_state_dict(torch.load("Training_Models/disc_A.pth",weights_only=True))
        disc_H.load_state_dict(torch.load("Training_Models/disc_B.pth",weights_only=True))
        gen_Z.load_state_dict(torch.load("Training_Models/gen_A.pth",weights_only=True))
        gen_H.load_state_dict(torch.load("Training_Models/gen_B.pth",weights_only=True))
    
    dataset = CGANDataset(
        root_A="Training/trainA",root_B="Training/trainB",transform=TRANSFORM
    )

    opt_disc = optim.Adam(
        list(disc_Z.parameters())+list(disc_H.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5,0.999)
    )
    opt_gen = optim.Adam(
        list(gen_Z.parameters())+list(gen_H.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5,0.999)
    )

    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    data_loader = DataLoader(
        dataset,
        batch_size = BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_OF_EPOCHS):
        H_reals = 0
        H_fakes = 0
        for idx, (zebra, horse) in enumerate(tqdm(data_loader)):
                zebra = zebra.to(DEVICE)
                horse = horse.to(DEVICE)

                # Train Discriminators H and Z
                with torch.cuda.amp.autocast():
                    fake_horse = gen_H(zebra)
                    D_H_real = disc_H(horse)
                    D_H_fake = disc_H(fake_horse.detach())
                    H_reals += D_H_real.mean().item()
                    H_fakes += D_H_fake.mean().item()
                    D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
                    D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
                    D_H_loss = D_H_real_loss + D_H_fake_loss

                    fake_zebra = gen_Z(horse)
                    D_Z_real = disc_Z(zebra)
                    D_Z_fake = disc_Z(fake_zebra.detach())
                    D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
                    D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
                    D_Z_loss = D_Z_real_loss + D_Z_fake_loss

                    # put it togethor
                    D_loss = (D_H_loss + D_Z_loss) / 2

                opt_disc.zero_grad()
                d_scaler.scale(D_loss).backward()
                d_scaler.step(opt_disc)
                d_scaler.update()

            # Train Generators H and Z
                with torch.cuda.amp.autocast():
                    # adversarial loss for both generators
                    D_H_fake = disc_H(fake_horse)
                    D_Z_fake = disc_Z(fake_zebra)
                    loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
                    loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

                    # cycle loss
                    cycle_zebra = gen_Z(fake_horse)
                    cycle_horse = gen_H(fake_zebra)
                    cycle_zebra_loss = l1(zebra, cycle_zebra)
                    cycle_horse_loss = l1(horse, cycle_horse)

                    # identity loss (remove these for efficiency if you set lambda_identity=0)
                    identity_zebra = gen_Z(zebra)
                    identity_horse = gen_H(horse)
                    identity_zebra_loss = l1(zebra, identity_zebra)
                    identity_horse_loss = l1(horse, identity_horse)

                    # add all togethor
                    G_loss = (
                        loss_G_Z
                        + loss_G_H
                        + cycle_zebra_loss * LAMBDA_CYCLE
                        + cycle_horse_loss * LAMBDA_CYCLE
                        + identity_horse_loss * LAMBDA_IDENTITY
                        + identity_zebra_loss * LAMBDA_IDENTITY
                    )

                opt_gen.zero_grad()
                g_scaler.scale(G_loss).backward()
                g_scaler.step(opt_gen)
                g_scaler.update()

        print(f"Epoch: [{epoch}/{NUM_OF_EPOCHS}] Critic Loss: {D_loss:.8f} | Gen loss: {G_loss:.8f}")
        torch.save(gen_Z.state_dict(),"Training_Models/gen_A.pth")
        torch.save(gen_H.state_dict(),"Training_Models/gen_B.pth")
        torch.save(disc_Z.state_dict(),"Training_Models/disc_A.pth")
        torch.save(disc_H.state_dict(),"Training_Models/disc_B.pth")
        
        

if __name__ == "__main__":
    main()
