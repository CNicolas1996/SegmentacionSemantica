import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds

class DataUtils:
    @staticmethod
    def download_model(model_url):
        """
        Descarga las librerias necesarias, como modelos  preentrenados o conjuntos de datos.

        Args:
        model_url (str): URL delmodelo a descargar.
        dataset_name (str): Nombre del dataset a descargar (opcional) **solo dela base de datos de TensorFlow**

        ....
        """
        #Aca se define la funcion para extraer elmodelo pre entrenado de keras
        tf.keras.utils.get_file(
            os.path.basename(model_url),
            model_url,
            cache_dir=os.path.join(os.getcwd(), 'models'),
            cache_subdir='pretrained',
            extract=True
        )

    @staticmethod
    def download_dataset(dataset_name=None, split='train', download=True):

        tfds.load(dataset_name, split=split, download=True) #terminar de estructurar la descarga de los datos en sus respectivos dirs

        

