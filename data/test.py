from RandomDataloader import RandomDataloader

rd = RandomDataloader(batch_size = 2, num_wavelengths = 5, num_layers = 4, shuffle = False, num_points = 4)
rd.load_data()

for i, batch in enumerate(rd):
    print(f"batch {i}:\n{batch}")