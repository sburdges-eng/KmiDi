#pragma once
/*
 * main_window.h - Qt Shell (Legacy)
 * ==================================
 *
 * CONNECTIONS (for Cursor Graph):
 * - Legacy UI: Qt-based experiments outside JUCE plugin surface
 * - Modern UI: src/ui/* + PluginEditor (primary product UI)
 *
 * Purpose: Minimal QMainWindow scaffold for older desktop prototypes.
 *
 * Features:
 * - setupUi, menus, toolbar hooks
 */

#include <QMainWindow>
#include <QWidget>

namespace kelly {

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override = default;

private:
    void setupUi();
    void createMenus();
    void createToolbar();
};

} // namespace kelly
