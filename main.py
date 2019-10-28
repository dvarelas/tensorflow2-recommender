from applications.matrix_factorization import mf_main, mf_config


if __name__ == '__main__':
    config = mf_config.config
    mf_main.main(config)


