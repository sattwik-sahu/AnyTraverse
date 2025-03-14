import base64
import logging
import time
from uuid import uuid4
from PIL import Image
import io

import roslibpy

# Configure logging
fmt = "%(asctime)s %(levelname)8s: %(message)s"
logging.basicConfig(format=fmt, level=logging.INFO)
log = logging.getLogger(__name__)

client = roslibpy.Ros(host="0.0.0.0", port=6969)

i = 0

def receive_image(msg):
    global i
    i += 1
    log.info("Received image seq=%d", i)
    base64_bytes = msg["data"].encode("ascii")
    image_bytes = base64.b64decode(base64_bytes)
    img = Image.open(io.BytesIO(image_bytes))
    img.save("data/img/recv/received-image-{}.{}".format(f"{i}".zfill(4), msg["format"]))
    # with open(
    #     "data/img/recv/received-image-{}.{}".format(f"{i}".zfill(4), msg["format"]), "wb"
    # ) as image_file:
    #     image_file.write(image_bytes)


subscriber = roslibpy.Topic(
    client, "/camera/image/compressed", "sensor_msgs/CompressedImage"
)
subscriber.subscribe(receive_image)

client.run_forever()
