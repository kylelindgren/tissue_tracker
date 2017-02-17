#include "mainwindow.hpp"
#include "../build-tissue_tracker-Desktop_Qt_5_7_0_GCC_64bit-Debug/ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow) {
    ui->setupUi(this);
}

MainWindow::~MainWindow() {
    delete ui;
}
