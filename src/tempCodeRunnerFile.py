
if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.info("divided by zero")
        raise CustomException(e, sys)
