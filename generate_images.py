import os
import pickle
import numpy as np
import tensorflow as tf
import PIL.Image

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string("checkpoint", None, "saved model to inference from")
tf.flags.DEFINE_string("output_dir", "output",
                       "output directory for generated images")


def main():
  # Initialize TensorFlow session.
  tf.InteractiveSession()

  # Import official CelebA-HQ networks.
  with open(FLAGS.checkpoint, 'rb') as file:
    G, D, Gs = pickle.load(file)

  # Generate latent vectors.
  latents = np.random.RandomState(1000).randn(
      1000, *Gs.input_shapes[0][1:])  # 1000 random latents
  latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341,
                     415]]  # hand-picked top-10

  # Generate dummy labels (not used by the official networks).
  labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

  # Run the generator to produce a set of images.
  images = Gs.run(latents, labels)

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
