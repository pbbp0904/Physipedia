from physipedia import Observer


def main():
    print("Starting program!")
    world = buildWorld()
    observations = Observer.getObservations(world)


if __name__ == "__main__":
    main()
