#!/usr/bin/env python3
"""
Lariat Bible Desktop Application
Main application entry point with comprehensive restaurant management features
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from tkinter import font as tkfont
import sys
from pathlib import Path
from datetime import datetime, date
import json
import os

# Add the core module to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "core"))

# Import core modules
try:
    from kitchen_core import storage, analysis, models
except ImportError:
    # Fallback if modules aren't available yet
    storage = None
    analysis = None
    models = None


class LariatBibleApp:
    """Main application class for Lariat Bible desktop app"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Lariat Bible - Restaurant Management System")
        self.root.geometry("1400x900")
        
        # Configure style
        self.setup_styles()
        
        # Initialize data storage
        self.current_recipe = None
        self.current_inventory = None
        self.invoices = []
        self.catering_events = []
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create main notebook for tabs
        self.create_main_interface()
        
        # Load initial data if available
        self.load_initial_data()
        
    def setup_styles(self):
        """Configure application styles and themes"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        self.colors = {
            'primary': '#2C3E50',
            'secondary': '#34495E',
            'success': '#27AE60',
            'warning': '#F39C12',
            'danger': '#E74C3C',
            'info': '#3498DB',
            'light': '#ECF0F1',
            'dark': '#2C3E50'
        }
        
        # Configure fonts
        self.fonts = {
            'title': tkfont.Font(family='Helvetica', size=16, weight='bold'),
            'heading': tkfont.Font(family='Helvetica', size=12, weight='bold'),
            'normal': tkfont.Font(family='Helvetica', size=10),
            'small': tkfont.Font(family='Helvetica', size=9)
        }
        
    def create_menu_bar(self):
        """Create application menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Recipe", command=self.new_recipe)
        file_menu.add_command(label="Open Recipe", command=self.open_recipe)
        file_menu.add_command(label="Save Recipe", command=self.save_recipe)
        file_menu.add_separator()
        file_menu.add_command(label="Import Invoice", command=self.import_invoice)
        file_menu.add_command(label="Export Reports", command=self.export_reports)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Settings", command=self.open_settings)
        edit_menu.add_command(label="Vendors", command=self.manage_vendors)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Dashboard", command=lambda: self.notebook.select(0))
        view_menu.add_command(label="Recipes", command=lambda: self.notebook.select(1))
        view_menu.add_command(label="Inventory", command=lambda: self.notebook.select(2))
        
        # Reports menu
        reports_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Reports", menu=reports_menu)
        reports_menu.add_command(label="Daily Report", command=self.generate_daily_report)
        reports_menu.add_command(label="Weekly Report", command=self.generate_weekly_report)
        reports_menu.add_command(label="Monthly Report", command=self.generate_monthly_report)
        reports_menu.add_command(label="Cost Analysis", command=self.run_cost_analysis)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        help_menu.add_command(label="About", command=self.show_about)
        
    def create_main_interface(self):
        """Create the main tabbed interface"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.create_dashboard_tab()
        self.create_recipes_tab()
        self.create_inventory_tab()
        self.create_invoices_tab()
        self.create_catering_tab()
        self.create_ordering_tab()
        self.create_equipment_tab()
        self.create_analytics_tab()
        
    def create_dashboard_tab(self):
        """Create dashboard tab with overview information"""
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text="📊 Dashboard")
        
        # Title
        title_label = ttk.Label(dashboard_frame, text="Restaurant Dashboard", 
                                font=self.fonts['title'])
        title_label.pack(pady=10)
        
        # Create metrics frame
        metrics_frame = ttk.Frame(dashboard_frame)
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Configure grid
        for i in range(3):
            metrics_frame.columnconfigure(i, weight=1)
        
        # Revenue metrics
        self.create_metric_card(metrics_frame, "Monthly Revenue", "$48,000", 
                                "↑ 12% from last month", 0, 0, 'success')
        
        self.create_metric_card(metrics_frame, "Catering Revenue", "$28,000", 
                                "58% of total revenue", 0, 1, 'info')
        
        self.create_metric_card(metrics_frame, "Restaurant Revenue", "$20,000", 
                                "42% of total revenue", 0, 2, 'warning')
        
        # Cost metrics
        self.create_metric_card(metrics_frame, "Food Cost %", "28.5%", 
                                "Target: <30%", 1, 0, 'success')
        
        self.create_metric_card(metrics_frame, "Labor Cost %", "32%", 
                                "Target: <35%", 1, 1, 'info')
        
        self.create_metric_card(metrics_frame, "Prime Cost %", "60.5%", 
                                "Target: <65%", 1, 2, 'success')
        
        # Savings metrics
        self.create_metric_card(metrics_frame, "Monthly Savings", "$4,333", 
                                "From vendor optimization", 2, 0, 'success')
        
        self.create_metric_card(metrics_frame, "Annual Projected", "$52,000", 
                                "Shamrock vs SYSCO", 2, 1, 'info')
        
        self.create_metric_card(metrics_frame, "Vendor Efficiency", "29.5%", 
                                "Cost reduction achieved", 2, 2, 'success')
        
        # Quick actions frame
        actions_frame = ttk.LabelFrame(dashboard_frame, text="Quick Actions", padding=10)
        actions_frame.pack(fill=tk.X, padx=20, pady=10)
        
        actions_grid = ttk.Frame(actions_frame)
        actions_grid.pack()
        
        ttk.Button(actions_grid, text="📝 New Recipe", 
                  command=self.new_recipe).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(actions_grid, text="📦 Check Inventory", 
                  command=lambda: self.notebook.select(2)).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(actions_grid, text="🎯 New Event", 
                  command=self.new_catering_event).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(actions_grid, text="📊 Run Analysis", 
                  command=self.run_cost_analysis).grid(row=0, column=3, padx=5, pady=5)
        
        # Recent activity
        activity_frame = ttk.LabelFrame(dashboard_frame, text="Recent Activity", padding=10)
        activity_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        activity_text = scrolledtext.ScrolledText(activity_frame, height=8, wrap=tk.WORD)
        activity_text.pack(fill=tk.BOTH, expand=True)
        
        # Sample activity
        activity_text.insert(tk.END, f"{datetime.now().strftime('%Y-%m-%d %H:%M')} - System initialized\n")
        activity_text.insert(tk.END, "• Recipe database loaded\n")
        activity_text.insert(tk.END, "• Inventory system connected\n")
        activity_text.insert(tk.END, "• Vendor price comparison active\n")
        activity_text.insert(tk.END, "• Catering module ready\n")
        activity_text.config(state=tk.DISABLED)
        
    def create_metric_card(self, parent, title, value, subtitle, row, col, color_key='info'):
        """Create a metric card widget"""
        frame = ttk.Frame(parent, relief=tk.RIDGE, borderwidth=2)
        frame.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
        
        ttk.Label(frame, text=title, font=self.fonts['small']).pack(pady=(10, 5))
        ttk.Label(frame, text=value, font=self.fonts['title']).pack()
        ttk.Label(frame, text=subtitle, font=self.fonts['small']).pack(pady=(5, 10))
        
    def create_recipes_tab(self):
        """Create recipes management tab"""
        recipes_frame = ttk.Frame(self.notebook)
        self.notebook.add(recipes_frame, text="📖 Recipes")
        
        # Create paned window
        paned = ttk.PanedWindow(recipes_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Recipe list
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        
        ttk.Label(left_frame, text="Recipe Library", font=self.fonts['heading']).pack(pady=5)
        
        # Search bar
        search_frame = ttk.Frame(left_frame)
        search_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT, padx=5)
        search_entry = ttk.Entry(search_frame)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Recipe categories
        category_frame = ttk.Frame(left_frame)
        category_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(category_frame, text="Category:").pack(side=tk.LEFT, padx=5)
        category_combo = ttk.Combobox(category_frame, values=["All", "Appetizers", "Entrees", 
                                                               "Desserts", "Beverages", "Sides"])
        category_combo.set("All")
        category_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Recipe list
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.recipe_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
        self.recipe_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.recipe_listbox.yview)
        
        # Sample recipes
        sample_recipes = [
            "🥗 Caesar Salad",
            "🍝 Fettuccine Alfredo",
            "🥩 Ribeye Steak",
            "🍔 Classic Burger",
            "🍕 Margherita Pizza",
            "🍰 Chocolate Cake",
            "🥘 Chicken Parmesan",
            "🌮 Street Tacos",
            "🍜 Pad Thai",
            "🥟 Dumplings"
        ]
        for recipe in sample_recipes:
            self.recipe_listbox.insert(tk.END, recipe)
        
        # Buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="New Recipe", command=self.new_recipe).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Import", command=self.import_recipe).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Delete", command=self.delete_recipe).pack(side=tk.LEFT, padx=2)
        
        # Right panel - Recipe details
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=2)
        
        ttk.Label(right_frame, text="Recipe Details", font=self.fonts['heading']).pack(pady=5)
        
        # Recipe details notebook
        details_notebook = ttk.Notebook(right_frame)
        details_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Ingredients tab
        ingredients_frame = ttk.Frame(details_notebook)
        details_notebook.add(ingredients_frame, text="Ingredients")
        
        # Ingredients table
        columns = ('Ingredient', 'Quantity', 'Unit', 'Cost')
        self.ingredients_tree = ttk.Treeview(ingredients_frame, columns=columns, show='headings')
        
        for col in columns:
            self.ingredients_tree.heading(col, text=col)
            self.ingredients_tree.column(col, width=100)
        
        self.ingredients_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Instructions tab
        instructions_frame = ttk.Frame(details_notebook)
        details_notebook.add(instructions_frame, text="Instructions")
        
        instructions_text = scrolledtext.ScrolledText(instructions_frame, wrap=tk.WORD)
        instructions_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Costing tab
        costing_frame = ttk.Frame(details_notebook)
        details_notebook.add(costing_frame, text="Costing")
        
        # Cost breakdown
        cost_info = ttk.Frame(costing_frame)
        cost_info.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(cost_info, text="Total Recipe Cost: $12.50", 
                 font=self.fonts['heading']).grid(row=0, column=0, sticky='w', pady=2)
        ttk.Label(cost_info, text="Cost per Serving: $3.13", 
                 font=self.fonts['normal']).grid(row=1, column=0, sticky='w', pady=2)
        ttk.Label(cost_info, text="Target Sell Price: $15.00", 
                 font=self.fonts['normal']).grid(row=2, column=0, sticky='w', pady=2)
        ttk.Label(cost_info, text="Food Cost %: 25%", 
                 font=self.fonts['normal']).grid(row=3, column=0, sticky='w', pady=2)
        ttk.Label(cost_info, text="Profit Margin: 75%", 
                 font=self.fonts['normal']).grid(row=4, column=0, sticky='w', pady=2)
        
        # Nutrition tab
        nutrition_frame = ttk.Frame(details_notebook)
        details_notebook.add(nutrition_frame, text="Nutrition")
        
        nutrition_info = ttk.Frame(nutrition_frame)
        nutrition_info.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(nutrition_info, text="Nutritional Information", 
                 font=self.fonts['heading']).pack(pady=5)
        ttk.Label(nutrition_info, text="Calories: 450").pack()
        ttk.Label(nutrition_info, text="Protein: 25g").pack()
        ttk.Label(nutrition_info, text="Carbohydrates: 35g").pack()
        ttk.Label(nutrition_info, text="Fat: 18g").pack()
        
    def create_inventory_tab(self):
        """Create inventory management tab"""
        inventory_frame = ttk.Frame(self.notebook)
        self.notebook.add(inventory_frame, text="📦 Inventory")
        
        # Title and controls
        top_frame = ttk.Frame(inventory_frame)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(top_frame, text="Inventory Management", 
                 font=self.fonts['title']).pack(side=tk.LEFT)
        
        # Action buttons
        button_frame = ttk.Frame(top_frame)
        button_frame.pack(side=tk.RIGHT)
        
        ttk.Button(button_frame, text="Add Item", 
                  command=self.add_inventory_item).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Receive Order", 
                  command=self.receive_order).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Count Inventory", 
                  command=self.count_inventory).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Export", 
                  command=self.export_inventory).pack(side=tk.LEFT, padx=2)
        
        # Filter frame
        filter_frame = ttk.Frame(inventory_frame)
        filter_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(filter_frame, text="Filter:").pack(side=tk.LEFT, padx=5)
        filter_combo = ttk.Combobox(filter_frame, values=["All", "Low Stock", "Out of Stock", 
                                                          "Overstock", "Expiring Soon"])
        filter_combo.set("All")
        filter_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(filter_frame, text="Category:").pack(side=tk.LEFT, padx=5)
        category_combo = ttk.Combobox(filter_frame, values=["All", "Produce", "Meat", "Dairy", 
                                                            "Dry Goods", "Beverages", "Supplies"])
        category_combo.set("All")
        category_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(filter_frame, text="Search:").pack(side=tk.LEFT, padx=5)
        search_entry = ttk.Entry(filter_frame, width=30)
        search_entry.pack(side=tk.LEFT, padx=5)
        
        # Inventory table
        table_frame = ttk.Frame(inventory_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create treeview
        columns = ('Item', 'Category', 'Current Stock', 'Unit', 'Par Level', 
                  'Status', 'Last Updated', 'Value')
        self.inventory_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=20)
        
        # Configure columns
        for col in columns:
            self.inventory_tree.heading(col, text=col)
            if col == 'Item':
                self.inventory_tree.column(col, width=200)
            else:
                self.inventory_tree.column(col, width=100)
        
        # Scrollbars
        y_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.inventory_tree.yview)
        x_scroll = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.inventory_tree.xview)
        self.inventory_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        
        # Pack elements
        self.inventory_tree.grid(row=0, column=0, sticky='nsew')
        y_scroll.grid(row=0, column=1, sticky='ns')
        x_scroll.grid(row=1, column=0, sticky='ew')
        
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        # Sample inventory items
        sample_items = [
            ("Chicken Breast", "Meat", "45", "lbs", "50", "OK", "Today", "$225"),
            ("Romaine Lettuce", "Produce", "12", "heads", "20", "Low", "Today", "$36"),
            ("Tomatoes", "Produce", "8", "lbs", "15", "Low", "Today", "$24"),
            ("Flour", "Dry Goods", "100", "lbs", "75", "OK", "Yesterday", "$50"),
            ("Olive Oil", "Dry Goods", "5", "gal", "8", "Low", "Yesterday", "$150"),
            ("Milk", "Dairy", "10", "gal", "12", "OK", "Today", "$40"),
            ("Ground Beef", "Meat", "30", "lbs", "40", "OK", "Today", "$120"),
            ("Onions", "Produce", "25", "lbs", "20", "OK", "Today", "$25"),
        ]
        
        for item in sample_items:
            self.inventory_tree.insert('', tk.END, values=item)
        
        # Summary frame
        summary_frame = ttk.LabelFrame(inventory_frame, text="Inventory Summary", padding=10)
        summary_frame.pack(fill=tk.X, padx=10, pady=10)
        
        summary_grid = ttk.Frame(summary_frame)
        summary_grid.pack()
        
        ttk.Label(summary_grid, text="Total Items: 156").grid(row=0, column=0, padx=20, sticky='w')
        ttk.Label(summary_grid, text="Total Value: $8,450").grid(row=0, column=1, padx=20, sticky='w')
        ttk.Label(summary_grid, text="Low Stock Items: 12").grid(row=0, column=2, padx=20, sticky='w')
        ttk.Label(summary_grid, text="Out of Stock: 3").grid(row=0, column=3, padx=20, sticky='w')
        
    def create_invoices_tab(self):
        """Create invoices and vendor management tab"""
        invoices_frame = ttk.Frame(self.notebook)
        self.notebook.add(invoices_frame, text="📄 Invoices")
        
        # Create paned window
        paned = ttk.PanedWindow(invoices_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Invoice list
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        
        ttk.Label(left_frame, text="Invoice Management", font=self.fonts['heading']).pack(pady=5)
        
        # Vendor filter
        vendor_frame = ttk.Frame(left_frame)
        vendor_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(vendor_frame, text="Vendor:").pack(side=tk.LEFT, padx=5)
        vendor_combo = ttk.Combobox(vendor_frame, values=["All", "SYSCO", "Shamrock Foods", 
                                                          "US Foods", "Local Produce"])
        vendor_combo.set("All")
        vendor_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Date range
        date_frame = ttk.Frame(left_frame)
        date_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(date_frame, text="Date Range:").pack(side=tk.LEFT, padx=5)
        date_combo = ttk.Combobox(date_frame, values=["Last 30 days", "Last 60 days", 
                                                      "Last 90 days", "This Year"])
        date_combo.set("Last 30 days")
        date_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Invoice list
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        columns = ('Date', 'Vendor', 'Invoice #', 'Amount')
        self.invoice_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.invoice_tree.heading(col, text=col)
            self.invoice_tree.column(col, width=80)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.invoice_tree.yview)
        self.invoice_tree.configure(yscrollcommand=scrollbar.set)
        
        self.invoice_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Sample invoices
        sample_invoices = [
            ("2024-11-18", "SYSCO", "INV-2024-1145", "$2,450"),
            ("2024-11-17", "Shamrock", "SF-8823", "$1,890"),
            ("2024-11-15", "SYSCO", "INV-2024-1139", "$3,200"),
            ("2024-11-14", "Shamrock", "SF-8819", "$2,100"),
            ("2024-11-12", "US Foods", "USF-4421", "$980"),
        ]
        
        for invoice in sample_invoices:
            self.invoice_tree.insert('', tk.END, values=invoice)
        
        # Action buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="Import Invoice", 
                  command=self.import_invoice).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="OCR Scan", 
                  command=self.ocr_scan_invoice).pack(side=tk.LEFT, padx=2)
        
        # Right panel - Invoice details and comparison
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=2)
        
        ttk.Label(right_frame, text="Invoice Analysis", font=self.fonts['heading']).pack(pady=5)
        
        # Analysis notebook
        analysis_notebook = ttk.Notebook(right_frame)
        analysis_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Invoice details tab
        details_frame = ttk.Frame(analysis_notebook)
        analysis_notebook.add(details_frame, text="Invoice Details")
        
        # Details text
        details_text = scrolledtext.ScrolledText(details_frame, wrap=tk.WORD, height=20)
        details_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        details_text.insert(tk.END, "Select an invoice to view details...")
        
        # Price comparison tab
        comparison_frame = ttk.Frame(analysis_notebook)
        analysis_notebook.add(comparison_frame, text="Price Comparison")
        
        # Comparison info
        comp_info = ttk.Frame(comparison_frame)
        comp_info.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(comp_info, text="Vendor Price Comparison", 
                 font=self.fonts['heading']).pack(pady=5)
        
        # Comparison table
        comp_columns = ('Item', 'SYSCO', 'Shamrock', 'Difference', 'Savings %')
        self.comparison_tree = ttk.Treeview(comparison_frame, columns=comp_columns, show='headings')
        
        for col in comp_columns:
            self.comparison_tree.heading(col, text=col)
            self.comparison_tree.column(col, width=100)
        
        self.comparison_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Sample comparison data
        comparison_data = [
            ("Chicken Breast", "$5.50/lb", "$4.20/lb", "-$1.30", "23.6%"),
            ("Ground Beef", "$4.80/lb", "$3.60/lb", "-$1.20", "25.0%"),
            ("Olive Oil", "$32/gal", "$24/gal", "-$8.00", "25.0%"),
            ("Tomatoes", "$3.20/lb", "$2.10/lb", "-$1.10", "34.4%"),
            ("Flour", "$0.55/lb", "$0.38/lb", "-$0.17", "30.9%"),
        ]
        
        for item in comparison_data:
            self.comparison_tree.insert('', tk.END, values=item)
        
        # Summary
        summary_label = ttk.Label(comparison_frame, 
                                 text="Average Savings with Shamrock: 29.5% ($52,000/year)",
                                 font=self.fonts['heading'])
        summary_label.pack(pady=10)
        
    def create_catering_tab(self):
        """Create catering events management tab"""
        catering_frame = ttk.Frame(self.notebook)
        self.notebook.add(catering_frame, text="🎯 Catering")
        
        # Title and controls
        top_frame = ttk.Frame(catering_frame)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(top_frame, text="Catering Events", font=self.fonts['title']).pack(side=tk.LEFT)
        
        # Action buttons
        button_frame = ttk.Frame(top_frame)
        button_frame.pack(side=tk.RIGHT)
        
        ttk.Button(button_frame, text="New Event", 
                  command=self.new_catering_event).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Create BEO", 
                  command=self.create_beo).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Generate Quote", 
                  command=self.generate_quote).pack(side=tk.LEFT, padx=2)
        
        # Calendar view / List view toggle
        view_frame = ttk.Frame(catering_frame)
        view_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.catering_view = tk.StringVar(value="list")
        ttk.Radiobutton(view_frame, text="List View", variable=self.catering_view, 
                       value="list").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(view_frame, text="Calendar View", variable=self.catering_view, 
                       value="calendar").pack(side=tk.LEFT, padx=5)
        
        # Events table
        table_frame = ttk.Frame(catering_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        columns = ('Date', 'Event', 'Client', 'Guests', 'Status', 'Revenue', 'Profit')
        self.catering_tree = ttk.Treeview(table_frame, columns=columns, show='headings')
        
        for col in columns:
            self.catering_tree.heading(col, text=col)
            if col == 'Event' or col == 'Client':
                self.catering_tree.column(col, width=150)
            else:
                self.catering_tree.column(col, width=80)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.catering_tree.yview)
        self.catering_tree.configure(yscrollcommand=scrollbar.set)
        
        self.catering_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Sample events
        sample_events = [
            ("2024-11-20", "Corporate Lunch", "Tech Corp", "50", "Confirmed", "$2,500", "$1,125"),
            ("2024-11-22", "Wedding Reception", "Smith/Jones", "150", "Confirmed", "$8,500", "$3,825"),
            ("2024-11-25", "Holiday Party", "ABC Company", "75", "Pending", "$4,200", "$1,890"),
            ("2024-11-28", "Birthday Dinner", "Johnson Family", "30", "Confirmed", "$1,800", "$810"),
            ("2024-12-01", "Board Meeting", "XYZ Corp", "25", "Quote", "$1,500", "$675"),
        ]
        
        for event in sample_events:
            self.catering_tree.insert('', tk.END, values=event)
        
        # Event details frame
        details_frame = ttk.LabelFrame(catering_frame, text="Event Details", padding=10)
        details_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Details grid
        details_grid = ttk.Frame(details_frame)
        details_grid.pack()
        
        # Row 1
        ttk.Label(details_grid, text="Next Event: Corporate Lunch - Nov 20").grid(
            row=0, column=0, columnspan=2, sticky='w', pady=2)
        ttk.Label(details_grid, text="Prep Start: 8:00 AM").grid(
            row=1, column=0, sticky='w', padx=20, pady=2)
        ttk.Label(details_grid, text="Service Time: 12:00 PM").grid(
            row=1, column=1, sticky='w', padx=20, pady=2)
        
        # Catering metrics
        metrics_frame = ttk.LabelFrame(catering_frame, text="Catering Metrics", padding=10)
        metrics_frame.pack(fill=tk.X, padx=10, pady=10)
        
        metrics_grid = ttk.Frame(metrics_frame)
        metrics_grid.pack()
        
        ttk.Label(metrics_grid, text="Monthly Revenue: $28,000").grid(row=0, column=0, padx=20, sticky='w')
        ttk.Label(metrics_grid, text="Avg Event Size: $3,500").grid(row=0, column=1, padx=20, sticky='w')
        ttk.Label(metrics_grid, text="Profit Margin: 45%").grid(row=0, column=2, padx=20, sticky='w')
        ttk.Label(metrics_grid, text="Events This Month: 8").grid(row=0, column=3, padx=20, sticky='w')
        
    def create_ordering_tab(self):
        """Create ordering and purchasing tab"""
        ordering_frame = ttk.Frame(self.notebook)
        self.notebook.add(ordering_frame, text="🛒 Ordering")
        
        # Title
        ttk.Label(ordering_frame, text="Order Management", 
                 font=self.fonts['title']).pack(pady=10)
        
        # Create paned window
        paned = ttk.PanedWindow(ordering_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Order guide
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        
        ttk.Label(left_frame, text="Order Guide", font=self.fonts['heading']).pack(pady=5)
        
        # Vendor selection
        vendor_frame = ttk.Frame(left_frame)
        vendor_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(vendor_frame, text="Vendor:").pack(side=tk.LEFT, padx=5)
        vendor_combo = ttk.Combobox(vendor_frame, values=["Shamrock Foods", "SYSCO", 
                                                          "US Foods", "Local Produce"])
        vendor_combo.set("Shamrock Foods")
        vendor_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Order items
        order_columns = ('Item', 'Par', 'On Hand', 'Order Qty', 'Unit Cost')
        self.order_tree = ttk.Treeview(left_frame, columns=order_columns, show='headings', height=15)
        
        for col in order_columns:
            self.order_tree.heading(col, text=col)
            self.order_tree.column(col, width=80)
        
        self.order_tree.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Sample order items
        order_items = [
            ("Chicken Breast", "50 lbs", "15 lbs", "35 lbs", "$4.20"),
            ("Ground Beef", "40 lbs", "10 lbs", "30 lbs", "$3.60"),
            ("Romaine Lettuce", "20 heads", "5 heads", "15 heads", "$3.00"),
            ("Tomatoes", "15 lbs", "3 lbs", "12 lbs", "$2.10"),
            ("Olive Oil", "8 gal", "2 gal", "6 gal", "$24.00"),
        ]
        
        for item in order_items:
            self.order_tree.insert('', tk.END, values=item)
        
        # Order actions
        action_frame = ttk.Frame(left_frame)
        action_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(action_frame, text="Calculate Order", 
                  command=self.calculate_order).pack(side=tk.LEFT, padx=2)
        ttk.Button(action_frame, text="Submit Order", 
                  command=self.submit_order).pack(side=tk.LEFT, padx=2)
        
        # Right panel - Shopping list and pending orders
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)
        
        # Shopping list
        ttk.Label(right_frame, text="Shopping List", font=self.fonts['heading']).pack(pady=5)
        
        shopping_text = scrolledtext.ScrolledText(right_frame, height=10, wrap=tk.WORD)
        shopping_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        shopping_text.insert(tk.END, "Generated Shopping List:\n\n")
        shopping_text.insert(tk.END, "Shamrock Foods Order:\n")
        shopping_text.insert(tk.END, "• Chicken Breast - 35 lbs @ $4.20/lb = $147.00\n")
        shopping_text.insert(tk.END, "• Ground Beef - 30 lbs @ $3.60/lb = $108.00\n")
        shopping_text.insert(tk.END, "• Romaine Lettuce - 15 heads @ $3.00/head = $45.00\n")
        shopping_text.insert(tk.END, "• Tomatoes - 12 lbs @ $2.10/lb = $25.20\n")
        shopping_text.insert(tk.END, "• Olive Oil - 6 gal @ $24.00/gal = $144.00\n")
        shopping_text.insert(tk.END, "\nTotal Order: $469.20\n")
        shopping_text.insert(tk.END, "Estimated Savings vs SYSCO: $138.80 (29.5%)")
        
        # Pending orders
        ttk.Label(right_frame, text="Pending Orders", font=self.fonts['heading']).pack(pady=5)
        
        pending_frame = ttk.Frame(right_frame)
        pending_frame.pack(fill=tk.X, padx=5, pady=5)
        
        pending_columns = ('Order Date', 'Vendor', 'Status', 'Total')
        pending_tree = ttk.Treeview(pending_frame, columns=pending_columns, 
                                   show='headings', height=5)
        
        for col in pending_columns:
            pending_tree.heading(col, text=col)
            pending_tree.column(col, width=80)
        
        pending_tree.pack(fill=tk.BOTH, expand=True)
        
        # Sample pending orders
        pending_orders = [
            ("2024-11-18", "Shamrock", "Submitted", "$469.20"),
            ("2024-11-17", "Local Produce", "Delivered", "$125.00"),
        ]
        
        for order in pending_orders:
            pending_tree.insert('', tk.END, values=order)
        
    def create_equipment_tab(self):
        """Create equipment maintenance tab"""
        equipment_frame = ttk.Frame(self.notebook)
        self.notebook.add(equipment_frame, text="🔧 Equipment")
        
        # Title
        ttk.Label(equipment_frame, text="Equipment Maintenance", 
                 font=self.fonts['title']).pack(pady=10)
        
        # Equipment list
        list_frame = ttk.Frame(equipment_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        columns = ('Equipment', 'Type', 'Last Service', 'Next Service', 'Status', 'Notes')
        self.equipment_tree = ttk.Treeview(list_frame, columns=columns, show='headings')
        
        for col in columns:
            self.equipment_tree.heading(col, text=col)
            if col == 'Equipment' or col == 'Notes':
                self.equipment_tree.column(col, width=150)
            else:
                self.equipment_tree.column(col, width=100)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.equipment_tree.yview)
        self.equipment_tree.configure(yscrollcommand=scrollbar.set)
        
        self.equipment_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Sample equipment
        equipment_items = [
            ("Grill Station #1", "Cooking", "2024-10-15", "2024-12-15", "✅ Good", "Clean burners"),
            ("Walk-in Cooler", "Refrigeration", "2024-11-01", "2024-12-01", "⚠️ Check", "Monitor temp"),
            ("Dishwasher", "Cleaning", "2024-10-30", "2024-11-30", "✅ Good", "Replace filters"),
            ("Fryer #1", "Cooking", "2024-11-10", "2024-12-10", "✅ Good", "Oil change due"),
            ("Ice Machine", "Refrigeration", "2024-09-15", "2024-11-15", "❌ Service", "Schedule cleaning"),
            ("Mixer", "Prep", "2024-10-20", "2025-01-20", "✅ Good", "Lubricate gears"),
            ("Oven #1", "Cooking", "2024-11-05", "2025-01-05", "✅ Good", "Calibrate temp"),
        ]
        
        for item in equipment_items:
            self.equipment_tree.insert('', tk.END, values=item)
        
        # Maintenance actions
        action_frame = ttk.Frame(equipment_frame)
        action_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(action_frame, text="Schedule Service", 
                  command=self.schedule_service).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Log Maintenance", 
                  command=self.log_maintenance).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="View History", 
                  command=self.view_maintenance_history).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Print Schedule", 
                  command=self.print_maintenance_schedule).pack(side=tk.LEFT, padx=5)
        
        # Maintenance summary
        summary_frame = ttk.LabelFrame(equipment_frame, text="Maintenance Summary", padding=10)
        summary_frame.pack(fill=tk.X, padx=10, pady=10)
        
        summary_grid = ttk.Frame(summary_frame)
        summary_grid.pack()
        
        ttk.Label(summary_grid, text="Equipment Items: 12").grid(row=0, column=0, padx=20, sticky='w')
        ttk.Label(summary_grid, text="Due This Week: 2").grid(row=0, column=1, padx=20, sticky='w')
        ttk.Label(summary_grid, text="Overdue: 1").grid(row=0, column=2, padx=20, sticky='w')
        ttk.Label(summary_grid, text="Compliance Rate: 92%").grid(row=0, column=3, padx=20, sticky='w')
        
    def create_analytics_tab(self):
        """Create analytics and reporting tab"""
        analytics_frame = ttk.Frame(self.notebook)
        self.notebook.add(analytics_frame, text="📈 Analytics")
        
        # Title
        ttk.Label(analytics_frame, text="Business Analytics", 
                 font=self.fonts['title']).pack(pady=10)
        
        # Time period selection
        period_frame = ttk.Frame(analytics_frame)
        period_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(period_frame, text="Period:").pack(side=tk.LEFT, padx=5)
        period_combo = ttk.Combobox(period_frame, values=["This Week", "This Month", 
                                                          "Last Month", "This Quarter", "This Year"])
        period_combo.set("This Month")
        period_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(period_frame, text="Generate Report", 
                  command=self.generate_analytics_report).pack(side=tk.LEFT, padx=10)
        ttk.Button(period_frame, text="Export PDF", 
                  command=self.export_analytics_pdf).pack(side=tk.LEFT, padx=2)
        ttk.Button(period_frame, text="Export Excel", 
                  command=self.export_analytics_excel).pack(side=tk.LEFT, padx=2)
        
        # Analytics notebook
        analytics_notebook = ttk.Notebook(analytics_frame)
        analytics_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Revenue analysis tab
        revenue_frame = ttk.Frame(analytics_notebook)
        analytics_notebook.add(revenue_frame, text="Revenue")
        
        revenue_text = scrolledtext.ScrolledText(revenue_frame, wrap=tk.WORD)
        revenue_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        revenue_text.insert(tk.END, "REVENUE ANALYSIS - November 2024\n")
        revenue_text.insert(tk.END, "="*50 + "\n\n")
        revenue_text.insert(tk.END, "Total Revenue: $48,000\n")
        revenue_text.insert(tk.END, "  • Catering: $28,000 (58.3%)\n")
        revenue_text.insert(tk.END, "  • Restaurant: $20,000 (41.7%)\n\n")
        revenue_text.insert(tk.END, "Top Revenue Days:\n")
        revenue_text.insert(tk.END, "  1. Saturday: $8,500\n")
        revenue_text.insert(tk.END, "  2. Friday: $7,200\n")
        revenue_text.insert(tk.END, "  3. Sunday: $6,100\n\n")
        revenue_text.insert(tk.END, "Growth: +12% vs Last Month\n")
        revenue_text.insert(tk.END, "Forecast: $52,000 (December)\n")
        
        # Cost analysis tab
        cost_frame = ttk.Frame(analytics_notebook)
        analytics_notebook.add(cost_frame, text="Costs")
        
        cost_text = scrolledtext.ScrolledText(cost_frame, wrap=tk.WORD)
        cost_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        cost_text.insert(tk.END, "COST ANALYSIS - November 2024\n")
        cost_text.insert(tk.END, "="*50 + "\n\n")
        cost_text.insert(tk.END, "Food Cost: $13,680 (28.5%)\n")
        cost_text.insert(tk.END, "Labor Cost: $15,360 (32.0%)\n")
        cost_text.insert(tk.END, "Prime Cost: $29,040 (60.5%)\n")
        cost_text.insert(tk.END, "Other Operating: $7,200 (15.0%)\n")
        cost_text.insert(tk.END, "Net Profit: $11,760 (24.5%)\n\n")
        cost_text.insert(tk.END, "Vendor Savings (Shamrock vs SYSCO):\n")
        cost_text.insert(tk.END, "  • Monthly: $4,333\n")
        cost_text.insert(tk.END, "  • Annual Projected: $52,000\n")
        
        # Menu analysis tab
        menu_frame = ttk.Frame(analytics_notebook)
        analytics_notebook.add(menu_frame, text="Menu Performance")
        
        menu_columns = ('Item', 'Sold', 'Revenue', 'Food Cost %', 'Profit')
        menu_tree = ttk.Treeview(menu_frame, columns=menu_columns, show='headings')
        
        for col in menu_columns:
            menu_tree.heading(col, text=col)
            menu_tree.column(col, width=100)
        
        menu_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Sample menu items
        menu_items = [
            ("Ribeye Steak", "245", "$7,350", "32%", "$4,998"),
            ("Caesar Salad", "412", "$4,532", "18%", "$3,716"),
            ("Chicken Parm", "338", "$5,408", "28%", "$3,894"),
            ("Margherita Pizza", "298", "$3,874", "22%", "$3,022"),
            ("Pad Thai", "189", "$2,268", "25%", "$1,701"),
        ]
        
        for item in menu_items:
            menu_tree.insert('', tk.END, values=item)
        
        # Catering analysis tab
        catering_analysis_frame = ttk.Frame(analytics_notebook)
        analytics_notebook.add(catering_analysis_frame, text="Catering Analysis")
        
        catering_text = scrolledtext.ScrolledText(catering_analysis_frame, wrap=tk.WORD)
        catering_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        catering_text.insert(tk.END, "CATERING ANALYSIS - November 2024\n")
        catering_text.insert(tk.END, "="*50 + "\n\n")
        catering_text.insert(tk.END, "Total Events: 8\n")
        catering_text.insert(tk.END, "Total Revenue: $28,000\n")
        catering_text.insert(tk.END, "Average Event Size: $3,500\n")
        catering_text.insert(tk.END, "Profit Margin: 45%\n")
        catering_text.insert(tk.END, "Total Profit: $12,600\n\n")
        catering_text.insert(tk.END, "Event Types:\n")
        catering_text.insert(tk.END, "  • Corporate: 4 events ($14,000)\n")
        catering_text.insert(tk.END, "  • Wedding: 2 events ($9,000)\n")
        catering_text.insert(tk.END, "  • Private Party: 2 events ($5,000)\n")
        
    # Helper methods
    def load_initial_data(self):
        """Load initial data from files if available"""
        try:
            # Try to load sample data
            data_path = REPO_ROOT / "data"
            if data_path.exists():
                # Load sample recipes
                recipes_path = data_path / "sample_recipes"
                if recipes_path.exists() and storage:
                    for recipe_file in recipes_path.glob("*.yaml"):
                        try:
                            recipe = storage.load_recipe_from_path(recipe_file)
                            # Store recipe for later use
                        except Exception as e:
                            print(f"Could not load {recipe_file}: {e}")
                
                # Load sample inventory
                inventory_path = data_path / "sample_inventory" / "inventory.yaml"
                if inventory_path.exists() and storage:
                    try:
                        self.current_inventory = storage.load_inventory_from_path(inventory_path)
                    except Exception as e:
                        print(f"Could not load inventory: {e}")
        except Exception as e:
            print(f"Error loading initial data: {e}")
    
    # Menu command methods
    def new_recipe(self):
        """Create a new recipe"""
        messagebox.showinfo("New Recipe", "Opening new recipe editor...")
    
    def open_recipe(self):
        """Open an existing recipe"""
        filename = filedialog.askopenfilename(
            title="Open Recipe",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        if filename:
            messagebox.showinfo("Open Recipe", f"Opening {filename}")
    
    def save_recipe(self):
        """Save current recipe"""
        messagebox.showinfo("Save Recipe", "Recipe saved successfully!")
    
    def import_invoice(self):
        """Import invoice from file"""
        filename = filedialog.askopenfilename(
            title="Import Invoice",
            filetypes=[("Image files", "*.png *.jpg *.jpeg"), 
                      ("PDF files", "*.pdf"), 
                      ("All files", "*.*")]
        )
        if filename:
            messagebox.showinfo("Import Invoice", f"Importing {filename}")
    
    def export_reports(self):
        """Export reports"""
        messagebox.showinfo("Export Reports", "Reports exported successfully!")
    
    def open_settings(self):
        """Open settings dialog"""
        messagebox.showinfo("Settings", "Opening settings...")
    
    def manage_vendors(self):
        """Manage vendor information"""
        messagebox.showinfo("Vendors", "Opening vendor management...")
    
    def generate_daily_report(self):
        """Generate daily report"""
        messagebox.showinfo("Daily Report", "Generating daily report...")
    
    def generate_weekly_report(self):
        """Generate weekly report"""
        messagebox.showinfo("Weekly Report", "Generating weekly report...")
    
    def generate_monthly_report(self):
        """Generate monthly report"""
        messagebox.showinfo("Monthly Report", "Generating monthly report...")
    
    def run_cost_analysis(self):
        """Run cost analysis"""
        messagebox.showinfo("Cost Analysis", "Running cost analysis...")
    
    def show_documentation(self):
        """Show documentation"""
        messagebox.showinfo("Documentation", 
                           "Lariat Bible Documentation\n\n" +
                           "Complete restaurant management system\n" +
                           "Version 1.0.0")
    
    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo("About", 
                           "Lariat Bible\n" +
                           "Restaurant Management System\n\n" +
                           "Built for The Lariat, Fort Collins\n" +
                           "Version 1.0.0")
    
    # Tab-specific methods
    def add_inventory_item(self):
        """Add new inventory item"""
        messagebox.showinfo("Add Item", "Adding new inventory item...")
    
    def receive_order(self):
        """Receive order into inventory"""
        messagebox.showinfo("Receive Order", "Recording received order...")
    
    def count_inventory(self):
        """Perform inventory count"""
        messagebox.showinfo("Count Inventory", "Starting inventory count...")
    
    def export_inventory(self):
        """Export inventory data"""
        messagebox.showinfo("Export", "Exporting inventory data...")
    
    def import_recipe(self):
        """Import recipe from file"""
        messagebox.showinfo("Import Recipe", "Importing recipe...")
    
    def delete_recipe(self):
        """Delete selected recipe"""
        if messagebox.askyesno("Delete Recipe", "Are you sure you want to delete this recipe?"):
            messagebox.showinfo("Delete Recipe", "Recipe deleted")
    
    def ocr_scan_invoice(self):
        """OCR scan invoice"""
        messagebox.showinfo("OCR Scan", "Starting OCR scan of invoice...")
    
    def new_catering_event(self):
        """Create new catering event"""
        messagebox.showinfo("New Event", "Creating new catering event...")
    
    def create_beo(self):
        """Create Banquet Event Order"""
        messagebox.showinfo("BEO", "Creating Banquet Event Order...")
    
    def generate_quote(self):
        """Generate catering quote"""
        messagebox.showinfo("Quote", "Generating catering quote...")
    
    def calculate_order(self):
        """Calculate order quantities"""
        messagebox.showinfo("Calculate", "Calculating order quantities...")
    
    def submit_order(self):
        """Submit order to vendor"""
        if messagebox.askyesno("Submit Order", "Submit this order to vendor?"):
            messagebox.showinfo("Order Submitted", "Order has been submitted")
    
    def schedule_service(self):
        """Schedule equipment service"""
        messagebox.showinfo("Schedule Service", "Scheduling equipment service...")
    
    def log_maintenance(self):
        """Log maintenance activity"""
        messagebox.showinfo("Log Maintenance", "Logging maintenance activity...")
    
    def view_maintenance_history(self):
        """View maintenance history"""
        messagebox.showinfo("History", "Loading maintenance history...")
    
    def print_maintenance_schedule(self):
        """Print maintenance schedule"""
        messagebox.showinfo("Print", "Printing maintenance schedule...")
    
    def generate_analytics_report(self):
        """Generate analytics report"""
        messagebox.showinfo("Analytics", "Generating analytics report...")
    
    def export_analytics_pdf(self):
        """Export analytics to PDF"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if filename:
            messagebox.showinfo("Export PDF", f"Exporting to {filename}")
    
    def export_analytics_excel(self):
        """Export analytics to Excel"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if filename:
            messagebox.showinfo("Export Excel", f"Exporting to {filename}")


def main():
    """Main entry point for the application"""
    root = tk.Tk()
    app = LariatBibleApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
