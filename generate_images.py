import os
import pickle
import numpy as np
import tensorflow as tf
import PIL.Image

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string("checkpoint", None, "saved model to inference from")
tf.flags.DEFINE_string("output_dir", "output",
                       "output directory for generated images")
tf.flags.DEFINE_integer("num_images", 100, "number of images to generate")

def main():
  # Initialize TensorFlow session.
  tf.InteractiveSession()

  # Import official CelebA-HQ networks.
  with open(FLAGS.checkpoint, 'rb') as file:
    G, D, Gs = pickle.load(file)

  # create output_dir if not exists
  if not os.path.isdir(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)

  # Generate latent vectors.
  latents = np.random.RandomState(1000).randn(FLAGS.num_images, *Gs.input_shapes[0][1:])
  # Generate dummy labels (not used by the official networks).
  labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])
  
  images = np.zeros([FLAGS.num_images] + Gs.output_shapes[0][1:])
  # Run the generator to produce a set of images.
  for i in range(FLAGS.num_images):
      images[i] = Gs.run(np.asarray([latents[i]]), np.asarray([labels[i]]))

  # Convert images to PIL-compatible format.
  images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(
      np.uint8)  # [-1,1] => [0,255]
  images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC

  # Save images as PNG.
  for idx in range(images.shape[0]):
    filename = os.path.join(FLAGS.output_dir, "img%d.png" % idx)
    PIL.Image.fromarray(images[idx], 'RGB').save(filename)


if __name__ == "__main__":
  main()
