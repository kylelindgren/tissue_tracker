#-------------------------------------------------
#
# Project created by QtCreator 2016-12-14T15:05:52
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = tissue_tracker
TEMPLATE = app

INCLUDEPATH += /usr/local/include/opencv
#INCLUDEPATH += /usr/local/include/Eigen
LIBS += -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui \
        -lopencv_videoio -lopencv_imgproc -lopencv_features2d -lopencv_xfeatures2d \
        -lopencv_flann -lopencv_calib3d
LIBS += -L/home/kylelindgren/Qt/5.7/gcc_64/lib -lQt5OpenGL
LIBS += -L/usr/lib

SOURCES += main.cpp\
    mainwindow.cpp \
    ssim.cpp \
    mc.cpp

HEADERS  += mainwindow.hpp \
    ssim.hpp \
    mc.hpp

FORMS += mainwindow.ui
