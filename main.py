import datastream as ds



CLIENTS = [
    {
        "host" : "localhost",
        "port" : 45345
    }
]

def main():
    running = True

    dataserver = ds.Dataserver(8, CLIENTS)

    while(running):
        try:
            print(running)

        except KeyboardInterrupt:
            print("Clossing on interupt")
            running = False


    dataserver.close()

    print("Done")

if __name__ == "__main__":
    main()