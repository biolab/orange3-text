Import Documents
================

Import text documents from folders.

**Inputs**

- None

**Outputs**

- Corpus: A collection of documents from the local machine.

**Import Documents** widget retrieves text files from folders and creates a corpus. The widget reads .txt, .docx, .odt, .pdf and .xml files. If a folder contains subfolders, they will be used as class labels.

![](images/Import-Documents-stamped.png)

1. Folder being loaded.
2. Load folder from a local machine.
3. Reload the data.
4. Number of documents retrieved.

If the widget cannot read the file for some reason, the file will be skipped. Files that were successfully retrieved will still be on the output.

Example
-------

To retrieve the data, select the folder icon on the right side of the widget. Select the folder you wish to turn into corpus. Once the loading is finished, you will see how many documents the widget retrieved. To inspect them, connect the widget to [Corpus Viewer](corpusviewer.md). We've used a set of Kennedy's speeches in a plain text format.

![](images/Import-Documents-Example1.png)

Now let us try it with subfolders. We have placed Kennedy's speeches in two folders - pre-1962 and post-1962. If I load the parent folder, these two subfolders will be used as class labels. Check the output of the widget in a **Data Table**.

![](images/Import-Documents-Example2.png)
