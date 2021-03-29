#%%
from configparser import ConfigParser


def get_connection_string(config_path):
    config = ConfigParser(allow_no_value=True)
    config.read(config_path)
    section = config["connection"]
    driver = section["drivername"]
    host = section["host"]
    port = section["port"]
    user = section["username"]
    database = section["database"]
    return f"{driver}://{user}@{host}:{port}/{database}"


if __name__ == "__main__":
    print(get_connection_string("conf/presto.ini"))
