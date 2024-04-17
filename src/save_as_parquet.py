import file_management as fm


def transform_to_parquet(filename: fm.Filenames):
    df = fm.read_csv(filename)
    df.blue = df.blue.astype(bool)
    df.dual_sim = df.dual_sim.astype(bool)
    df.four_g = df.four_g.astype(bool)
    df.three_g = df.three_g.astype(bool)
    df.wifi = df.wifi.astype(bool)
    df.touch_screen = df.touch_screen.astype(bool)
    fm.save_parquet(filename, df)

def main():
    transform_to_parquet(fm.Filenames.train)
    transform_to_parquet(fm.Filenames.test)


if __name__ == '__main__':
    main()
