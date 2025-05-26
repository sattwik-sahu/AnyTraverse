from __future__ import print_function
import roslibpy

client = roslibpy.Ros(host='0.0.0.0', port=6969)
client.run()

listener = roslibpy.Topic(client, '/chatter', 'std_msgs/String')
listener.subscribe(lambda message: print('Heard talking: ' + message['data']))

try:
    while True:
        pass
except KeyboardInterrupt:
    client.terminate()
