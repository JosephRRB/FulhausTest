# Classifying Furniture Images

This webapp uses a small Convolutional Neural Network to label images 
into three different classes: `Bed`,  `Chair` or `Sofa`. For each
query image, a barchart of probabilities as predicted by the model is 
displayed. The user can select a query image from the validation set, 
upload an image file, or directly take a picture from their camera. 

The webapp is hosted in ***Heroku***: 
https://shrouded-retreat-14087.herokuapp.com/ 
or alternatively on ***Streamlit***: 
https://josephrrb-fulhaustest-main-app-wj99rm.streamlit.app/

For every push on the master branch, ***Github*** will build a 
***Docker*** image and deploy it on ***Heroku*** 