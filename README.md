Personal study notes built with [bookdown](https://bookdown.org/home/about.html) book. Thanks to [Sean Kross](http://seankross.com/2016/11/17/How-to-Start-a-Bookdown-Book.html) for the bookdown setup guidance.

Steps to change this template in to a ready to use bookdown book:

0. install bookdown - install.packageS("bookdown")
1. edit _output.yml with a relevant title (currently set to 'Study Notes Template')
2. edit _bookdown.yml there are a few parts to note:
    - book_filename: what the PDF and epubs will be called
    - chapter_name: What gets appended to the start of each chapter heading, if any
    - repo: Github repo associated [OPTIONAL]
    - output_dir: if not set, this will default to book.  Github will render a folder called /docs as a website
    - rmd_files: what files and their sequence to be rendered in to the book.  If not set, all files in root will be           included [OPTIONAL]
    - new_session: if set to yes, each .Rmd is rendered in its own R session, otherwise all Rmd files in your book are         rednered in the same R session
3. edit Index.Rmd, particuarly the title, author, date, github-repo, url, and description.
      If desired, you can add a field called cover-image for the book, with a link to a .png
      
To build the book, you can use either the 'Build book'. icon in R studio or `bookdown::render_book("index.Rmd")`

The files/book will be rendered either 

1) by chapter title e.g. 01-Name, 02-Name, which is you wish to order your chapters in this way name the files 0N-,

2) sequence them by the order of the files in the _bookdown.yml rmd_files field, which allows re-ordering.

**View as a website**

On the github website, go in to the repo, then hit the settings (cog) icon at the top right.  Scroll down to the 'Github pages' section and choose the option master branch /docs folder as the source.


**Travis CI - automated build**

If desired, it is possible to build the book remotley using Travis Continous Integration (or CI for short).  3 files are needed, 2 of which can be left as is in most cases - .Rbuildignore and travis.yml - the third needs to include any packages needed/used to build the book i.e. they are used within your Rmd files for graphics, analytics or any other purposes.  This is the DESCRIPTION file and the import or remotes should list those packages.

These files are already setup in the folder, however it is neccessary to setup Travis to link and build this repo.  