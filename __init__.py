# -*- coding: utf-8 -*-
"""
/***************************************************************************
 acnet_gui
                                 A QGIS plugin
 acnet_gui
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                             -------------------
        begin                : 2023-08-07
        copyright            : (C) 2023 by zectre
        email                : 112312@sddsa
        git sha              : $Format:%H$
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load acnet_gui class from file acnet_gui.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .acnet_gui import acnet_gui
    return acnet_gui(iface)
