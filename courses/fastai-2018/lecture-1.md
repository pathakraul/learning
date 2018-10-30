* Set the batch size **bs** accordingly.
* **untar_data(url)** will download the data in ~/.fastai/data

* **get_image_extensions(path)** -> calls get_files(path, image_extensions) to return a list of Path

* Never forget to set a random seed.

* We have data present and we can convert it into data bunch using DataBunch subclass according to type of dataset. For images it will be **ImageDataBunch**
