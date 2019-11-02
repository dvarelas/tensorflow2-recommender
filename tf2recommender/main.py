from tf2recommender.applications.neural_cf import ncf_main, ncf_config


if __name__ == '__main__':
    config = ncf_config.config
    ncf_main.main(config)


