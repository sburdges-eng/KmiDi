"""
Main Window for The Lariat Bible Desktop Application
Comprehensive restaurant management system with all features
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import tkinter.font as tkFont
from datetime import datetime
import json
import os
from pathlib import Path

# Import feature modules
from features.recipe_manager import RecipeManager
from features.inventory_manager import InventoryManager
from features.invoice_processor import InvoiceProcessor
from features.catering_manager import CateringManager
from features.cost_analyzer import CostAnalyzer
from features.vendor_manager import VendorManager
from features.equipment_maintenance import EquipmentMaintenance
from features.order_guide import OrderGuide
from features.reports_dashboard import ReportsDashboard

class LariatBibleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("The Lariat Bible - Restaurant Management System")
        
        # Set window size and center
        self.setup_window()
        
        # Configure styles
        self.setup_styles()
        
        # Create menu bar
        self.create_menubar()
        
        # Create main interface
        self.create_main_interface()
        
        # Initialize data
        self.current_user = "Sean Burdges"
        self.restaurant_name = "The Lariat"
        
        # Load initial data
        self.load_initial_data()
    
    def setup_window(self):
        """Configure the main window size and position"""
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Set window size (90% of screen)
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.9)
        
        # Center window
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.root.geometry(f'{window_width}x{window_height}+{x}+{y}')
        self.root.minsize(1200, 700)
        
        # Configure grid weight
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
    
    def setup_styles(self):
        """Configure ttk styles for modern look"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        self.colors = {
            'primary': '#2C3E50',      # Dark blue-gray
            'secondary': '#3498DB',     # Bright blue
            'success': '#27AE60',       # Green
            'warning': '#F39C12',       # Orange
            'danger': '#E74C3C',        # Red
            'light': '#ECF0F1',         # Light gray
            'dark': '#34495E',          # Dark gray
            'white': '#FFFFFF',
            'bg': '#F5F6FA'            # Background
        }
        
        # Configure button styles
        style.configure('Primary.TButton',
                       background=self.colors['primary'],
                       foreground=self.colors['white'],
                       borderwidth=0,
                       focuscolor='none',
                       padding=(10, 8))
        style.map('Primary.TButton',
                 background=[('active', self.colors['dark'])])
        
        style.configure('Success.TButton',
                       background=self.colors['success'],
                       foreground=self.colors['white'],
                       borderwidth=0,
                       padding=(10, 8))
        
        style.configure('Warning.TButton',
                       background=self.colors['warning'],
                       foreground=self.colors['white'],
                       borderwidth=0,
                       padding=(10, 8))
        
        # Configure notebook (tabs) style
        style.configure('Main.TNotebook',
                       background=self.colors['bg'],
                       borderwidth=0)
        style.configure('Main.TNotebook.Tab',
                       padding=(20, 12),
                       background=self.colors['light'])
        style.map('Main.TNotebook.Tab',
                 background=[('selected', self.colors['white'])],
                 expand=[('selected', [1, 1, 1, 0])])
    
    def create_menubar(self):
        """Create the application menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Recipe", command=self.new_recipe)
        file_menu.add_command(label="Import Data", command=self.import_data)
        file_menu.add_command(label="Export Data", command=self.export_data)
        file_menu.add_separator()
        file_menu.add_command(label="Settings", command=self.open_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="Find", accelerator="Ctrl+F")
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Dashboard", command=lambda: self.notebook.select(0))
        view_menu.add_command(label="Recipes", command=lambda: self.notebook.select(1))
        view_menu.add_command(label="Inventory", command=lambda: self.notebook.select(2))
        view_menu.add_command(label="Invoices", command=lambda: self.notebook.select(3))
        
        # Reports menu
        reports_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Reports", menu=reports_menu)
        reports_menu.add_command(label="Daily Sales", command=self.generate_daily_report)
        reports_menu.add_command(label="Monthly P&L", command=self.generate_monthly_report)
        reports_menu.add_command(label="Vendor Analysis", command=self.generate_vendor_report)
        reports_menu.add_command(label="Catering Summary", command=self.generate_catering_report)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        help_menu.add_command(label="About", command=self.show_about)
    
    def create_main_interface(self):
        """Create the main tabbed interface"""
        # Create main container
        main_container = ttk.Frame(self.root, style='Main.TFrame')
        main_container.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        main_container.grid_rowconfigure(1, weight=1)
        main_container.grid_columnconfigure(0, weight=1)
        
        # Create header
        self.create_header(main_container)
        
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(main_container, style='Main.TNotebook')
        self.notebook.grid(row=1, column=0, sticky='nsew', pady=(10, 0))
        
        # Create tabs for each feature
        self.tabs = {}
        
        # Dashboard Tab
        self.tabs['dashboard'] = ttk.Frame(self.notebook)
        self.notebook.add(self.tabs['dashboard'], text='📊 Dashboard')
        self.create_dashboard_tab()
        
        # Recipe Management Tab
        self.tabs['recipes'] = ttk.Frame(self.notebook)
        self.notebook.add(self.tabs['recipes'], text='📖 Recipes')
        self.create_recipes_tab()
        
        # Inventory Management Tab
        self.tabs['inventory'] = ttk.Frame(self.notebook)
        self.notebook.add(self.tabs['inventory'], text='📦 Inventory')
        self.create_inventory_tab()
        
        # Invoice Processing Tab
        self.tabs['invoices'] = ttk.Frame(self.notebook)
        self.notebook.add(self.tabs['invoices'], text='📄 Invoices')
        self.create_invoices_tab()
        
        # Catering Management Tab
        self.tabs['catering'] = ttk.Frame(self.notebook)
        self.notebook.add(self.tabs['catering'], text='🎉 Catering')
        self.create_catering_tab()
        
        # Vendor Management Tab
        self.tabs['vendors'] = ttk.Frame(self.notebook)
        self.notebook.add(self.tabs['vendors'], text='🚚 Vendors')
        self.create_vendors_tab()
        
        # Cost Analysis Tab
        self.tabs['costs'] = ttk.Frame(self.notebook)
        self.notebook.add(self.tabs['costs'], text='💰 Cost Analysis')
        self.create_costs_tab()
        
        # Equipment Maintenance Tab
        self.tabs['equipment'] = ttk.Frame(self.notebook)
        self.notebook.add(self.tabs['equipment'], text='🔧 Equipment')
        self.create_equipment_tab()
        
        # Order Guide Tab
        self.tabs['orders'] = ttk.Frame(self.notebook)
        self.notebook.add(self.tabs['orders'], text='📋 Order Guide')
        self.create_orders_tab()
        
        # Reports Tab
        self.tabs['reports'] = ttk.Frame(self.notebook)
        self.notebook.add(self.tabs['reports'], text='📈 Reports')
        self.create_reports_tab()
        
        # Create status bar
        self.create_statusbar()
    
    def create_header(self, parent):
        """Create application header"""
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        header_frame.grid_columnconfigure(1, weight=1)
        
        # Logo/Title
        title_font = tkFont.Font(family="Helvetica", size=24, weight="bold")
        title_label = tk.Label(header_frame,
                               text="🍴 The Lariat Bible",
                               font=title_font,
                               fg=self.colors['primary'])
        title_label.grid(row=0, column=0, sticky='w')
        
        # Restaurant info
        info_frame = ttk.Frame(header_frame)
        info_frame.grid(row=0, column=1, sticky='e')
        
        restaurant_label = tk.Label(info_frame,
                                   text=f"Restaurant: {self.restaurant_name}",
                                   font=("Helvetica", 11))
        restaurant_label.grid(row=0, column=0, padx=(0, 20))
        
        user_label = tk.Label(info_frame,
                             text=f"User: {self.current_user}",
                             font=("Helvetica", 11))
        user_label.grid(row=0, column=1, padx=(0, 20))
        
        date_label = tk.Label(info_frame,
                             text=f"Date: {datetime.now().strftime('%B %d, %Y')}",
                             font=("Helvetica", 11))
        date_label.grid(row=0, column=2)
    
    def create_dashboard_tab(self):
        """Create the dashboard tab with overview widgets"""
        dashboard = self.tabs['dashboard']
        dashboard.grid_columnconfigure(0, weight=1)
        dashboard.grid_columnconfigure(1, weight=1)
        dashboard.grid_rowconfigure(1, weight=1)
        
        # Title
        title_font = tkFont.Font(family="Helvetica", size=18, weight="bold")
        title = tk.Label(dashboard, text="Dashboard Overview", font=title_font)
        title.grid(row=0, column=0, columnspan=2, pady=20)
        
        # Left column - Quick Stats
        left_frame = ttk.LabelFrame(dashboard, text="Quick Statistics", padding=20)
        left_frame.grid(row=1, column=0, sticky='nsew', padx=(20, 10), pady=10)
        
        stats = [
            ("Monthly Revenue", "$48,000", self.colors['success']),
            ("Restaurant Sales", "$20,000", self.colors['primary']),
            ("Catering Sales", "$28,000", self.colors['secondary']),
            ("Food Cost %", "31.5%", self.colors['warning']),
            ("Labor Cost %", "28.2%", self.colors['warning']),
            ("Active Recipes", "156", self.colors['primary']),
            ("Pending Orders", "12", self.colors['danger']),
            ("Low Stock Items", "8", self.colors['danger'])
        ]
        
        for i, (label, value, color) in enumerate(stats):
            stat_frame = tk.Frame(left_frame, bg='white', relief='solid', borderwidth=1)
            stat_frame.grid(row=i//2, column=i%2, padx=10, pady=10, sticky='ew')
            
            tk.Label(stat_frame, text=label, bg='white', font=("Helvetica", 10)).pack(pady=(10, 5))
            tk.Label(stat_frame, text=value, bg='white', fg=color,
                    font=("Helvetica", 16, "bold")).pack(pady=(0, 10))
        
        # Right column - Recent Activity
        right_frame = ttk.LabelFrame(dashboard, text="Recent Activity", padding=20)
        right_frame.grid(row=1, column=1, sticky='nsew', padx=(10, 20), pady=10)
        
        # Activity list
        activity_text = tk.Text(right_frame, height=15, width=50, wrap='word')
        activity_text.pack(fill='both', expand=True)
        
        activities = [
            "✅ Invoice from Shamrock Foods processed - $3,245.67",
            "📦 Inventory count completed for Walk-in Cooler",
            "🎉 Catering event booked - Johnson Wedding (150 guests)",
            "⚠️ Low stock alert: Prime Rib (2 units remaining)",
            "📝 Recipe updated: Hickory Smoked Brisket",
            "💰 Cost analysis completed - 29.5% savings identified",
            "🔧 Maintenance scheduled: Convection Oven #2",
            "📊 Weekly P&L report generated",
            "✅ Order placed with SYSCO - $2,156.43",
            "📖 New recipe added: Jalapeño Cornbread"
        ]
        
        for activity in activities:
            activity_text.insert('end', f"• {activity}\n\n")
        activity_text.config(state='disabled')
        
        # Bottom row - Quick Actions
        actions_frame = ttk.LabelFrame(dashboard, text="Quick Actions", padding=20)
        actions_frame.grid(row=2, column=0, columnspan=2, sticky='ew', padx=20, pady=10)
        
        buttons = [
            ("New Recipe", self.new_recipe, 'Success.TButton'),
            ("Process Invoice", self.process_invoice, 'Primary.TButton'),
            ("Count Inventory", self.count_inventory, 'Primary.TButton'),
            ("Create Order", self.create_order, 'Primary.TButton'),
            ("Book Catering", self.book_catering, 'Success.TButton'),
            ("Generate Report", self.generate_report, 'Warning.TButton')
        ]
        
        for i, (text, command, style) in enumerate(buttons):
            btn = ttk.Button(actions_frame, text=text, command=command, style=style)
            btn.grid(row=0, column=i, padx=10, pady=5)
    
    def create_recipes_tab(self):
        """Create the recipes management tab"""
        recipes = self.tabs['recipes']
        recipes.grid_columnconfigure(0, weight=1)
        recipes.grid_rowconfigure(1, weight=1)
        
        # Toolbar
        toolbar = ttk.Frame(recipes)
        toolbar.grid(row=0, column=0, sticky='ew', padx=20, pady=10)
        
        ttk.Button(toolbar, text="➕ New Recipe", command=self.new_recipe,
                  style='Success.TButton').pack(side='left', padx=5)
        ttk.Button(toolbar, text="📝 Edit", command=self.edit_recipe,
                  style='Primary.TButton').pack(side='left', padx=5)
        ttk.Button(toolbar, text="🗑️ Delete", command=self.delete_recipe,
                  style='Warning.TButton').pack(side='left', padx=5)
        ttk.Button(toolbar, text="📋 Copy", command=self.copy_recipe).pack(side='left', padx=5)
        ttk.Button(toolbar, text="💰 Calculate Cost", command=self.calculate_recipe_cost).pack(side='left', padx=5)
        ttk.Button(toolbar, text="🖨️ Print", command=self.print_recipe).pack(side='left', padx=5)
        
        # Search bar
        search_frame = ttk.Frame(toolbar)
        search_frame.pack(side='right', padx=5)
        ttk.Label(search_frame, text="Search:").pack(side='left', padx=5)
        self.recipe_search = ttk.Entry(search_frame, width=30)
        self.recipe_search.pack(side='left')
        
        # Main content area with treeview
        content_frame = ttk.Frame(recipes)
        content_frame.grid(row=1, column=0, sticky='nsew', padx=20, pady=10)
        content_frame.grid_columnconfigure(0, weight=3)
        content_frame.grid_columnconfigure(1, weight=2)
        content_frame.grid_rowconfigure(0, weight=1)
        
        # Recipe list (left side)
        list_frame = ttk.LabelFrame(content_frame, text="Recipe List")
        list_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        
        # Create treeview with scrollbars
        tree_frame = ttk.Frame(list_frame)
        tree_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.recipe_tree = ttk.Treeview(tree_frame,
                                        columns=('Category', 'Cost', 'Margin', 'Prep Time'),
                                        show='tree headings',
                                        selectmode='browse')
        
        # Configure columns
        self.recipe_tree.heading('#0', text='Recipe Name')
        self.recipe_tree.heading('Category', text='Category')
        self.recipe_tree.heading('Cost', text='Cost')
        self.recipe_tree.heading('Margin', text='Margin %')
        self.recipe_tree.heading('Prep Time', text='Prep Time')
        
        self.recipe_tree.column('#0', width=250)
        self.recipe_tree.column('Category', width=150)
        self.recipe_tree.column('Cost', width=100)
        self.recipe_tree.column('Margin', width=100)
        self.recipe_tree.column('Prep Time', width=100)
        
        # Add sample recipes
        recipes_data = [
            ("Hickory Smoked Brisket", "Entrees", "$12.45", "68%", "4 hours"),
            ("Mac & Cheese", "Sides", "$2.85", "72%", "45 min"),
            ("Caesar Salad", "Salads", "$3.20", "75%", "15 min"),
            ("BBQ Ribs", "Entrees", "$8.75", "65%", "3 hours"),
            ("Coleslaw", "Sides", "$1.50", "80%", "20 min"),
            ("Jalapeño Cornbread", "Sides", "$1.25", "78%", "30 min"),
            ("Pulled Pork Sandwich", "Sandwiches", "$4.50", "70%", "6 hours"),
            ("Baked Beans", "Sides", "$1.75", "76%", "2 hours")
        ]
        
        for recipe in recipes_data:
            self.recipe_tree.insert('', 'end', text=recipe[0], values=recipe[1:])
        
        # Add scrollbars
        v_scroll = ttk.Scrollbar(tree_frame, orient='vertical', command=self.recipe_tree.yview)
        h_scroll = ttk.Scrollbar(tree_frame, orient='horizontal', command=self.recipe_tree.xview)
        self.recipe_tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        self.recipe_tree.grid(row=0, column=0, sticky='nsew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        h_scroll.grid(row=1, column=0, sticky='ew')
        
        tree_frame.grid_columnconfigure(0, weight=1)
        tree_frame.grid_rowconfigure(0, weight=1)
        
        # Recipe details (right side)
        details_frame = ttk.LabelFrame(content_frame, text="Recipe Details")
        details_frame.grid(row=0, column=1, sticky='nsew')
        
        details_text = tk.Text(details_frame, wrap='word', padx=10, pady=10)
        details_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        details_text.insert('1.0', """Recipe: Hickory Smoked Brisket
Category: Entrees
Serves: 10 portions
Prep Time: 4 hours
Cook Time: 12 hours

INGREDIENTS:
• 12 lb Beef Brisket
• 1/2 cup BBQ Rub
• 1/4 cup Black Pepper
• 1/4 cup Kosher Salt
• 2 tbsp Garlic Powder
• 2 tbsp Onion Powder
• Wood chips (hickory)

INSTRUCTIONS:
1. Trim brisket leaving 1/4" fat cap
2. Apply rub generously on all sides
3. Let rest at room temperature for 1 hour
4. Smoke at 225°F for 12-14 hours
5. Wrap in butcher paper at 165°F internal
6. Continue until 203°F internal temp
7. Rest for 1 hour before slicing

COST BREAKDOWN:
Brisket: $96.00
Seasonings: $3.45
Wood/Fuel: $8.00
Labor (4 hrs): $45.00
Total Cost: $152.45
Cost per portion: $15.25
Selling price: $47.50
Margin: 68%""")
        details_text.config(state='disabled')
    
    def create_inventory_tab(self):
        """Create the inventory management tab"""
        inventory = self.tabs['inventory']
        inventory.grid_columnconfigure(0, weight=1)
        inventory.grid_rowconfigure(1, weight=1)
        
        # Toolbar
        toolbar = ttk.Frame(inventory)
        toolbar.grid(row=0, column=0, sticky='ew', padx=20, pady=10)
        
        ttk.Button(toolbar, text="➕ Add Item", command=self.add_inventory,
                  style='Success.TButton').pack(side='left', padx=5)
        ttk.Button(toolbar, text="📝 Count", command=self.count_inventory,
                  style='Primary.TButton').pack(side='left', padx=5)
        ttk.Button(toolbar, text="📦 Receive", command=self.receive_inventory).pack(side='left', padx=5)
        ttk.Button(toolbar, text="⚠️ Set Par Levels", command=self.set_par_levels).pack(side='left', padx=5)
        ttk.Button(toolbar, text="📊 Usage Report", command=self.inventory_usage_report).pack(side='left', padx=5)
        
        # Filter options
        filter_frame = ttk.Frame(toolbar)
        filter_frame.pack(side='right', padx=5)
        
        ttk.Label(filter_frame, text="Filter:").pack(side='left', padx=5)
        self.inventory_filter = ttk.Combobox(filter_frame, width=20,
                                             values=['All Items', 'Low Stock', 'Proteins', 'Produce',
                                                    'Dairy', 'Dry Goods', 'Beverages'])
        self.inventory_filter.set('All Items')
        self.inventory_filter.pack(side='left')
        self.inventory_filter.bind('<<ComboboxSelected>>', self.on_inventory_filter_change)
        
        # Main inventory display
        inv_frame = ttk.Frame(inventory)
        inv_frame.grid(row=1, column=0, sticky='nsew', padx=20, pady=10)
        
        # Create treeview for inventory
        columns = ('Category', 'Quantity', 'Unit', 'Par Level', 'Cost', 'Value', 'Vendor', 'Status')
        self.inventory_tree = ttk.Treeview(inv_frame, columns=columns, show='tree headings')
        
        # Configure columns
        self.inventory_tree.heading('#0', text='Item Name')
        for col in columns:
            self.inventory_tree.heading(col, text=col)
        
        self.inventory_tree.column('#0', width=200)
        self.inventory_tree.column('Category', width=100)
        self.inventory_tree.column('Quantity', width=80)
        self.inventory_tree.column('Unit', width=80)
        self.inventory_tree.column('Par Level', width=80)
        self.inventory_tree.column('Cost', width=80)
        self.inventory_tree.column('Value', width=100)
        self.inventory_tree.column('Vendor', width=120)
        self.inventory_tree.column('Status', width=100)
        
        # Add sample inventory items
        inventory_items = [
            ("Beef Brisket", "Proteins", "8", "lb", "20", "$8.00", "$64.00", "Shamrock", "⚠️ Low"),
            ("Pork Shoulder", "Proteins", "45", "lb", "40", "$3.50", "$157.50", "SYSCO", "✅ Good"),
            ("Chicken Breast", "Proteins", "30", "lb", "25", "$4.25", "$127.50", "Shamrock", "✅ Good"),
            ("Romaine Lettuce", "Produce", "12", "head", "20", "$2.00", "$24.00", "Local Farm", "⚠️ Low"),
            ("Tomatoes", "Produce", "25", "lb", "30", "$1.50", "$37.50", "Local Farm", "✅ Good"),
            ("Cheddar Cheese", "Dairy", "15", "lb", "20", "$5.50", "$82.50", "SYSCO", "✅ Good"),
            ("Heavy Cream", "Dairy", "8", "qt", "12", "$4.00", "$32.00", "Shamrock", "⚠️ Low"),
            ("BBQ Sauce", "Condiments", "6", "gal", "10", "$12.00", "$72.00", "Shamrock", "⚠️ Low"),
            ("All Purpose Flour", "Dry Goods", "50", "lb", "40", "$0.75", "$37.50", "SYSCO", "✅ Good")
        ]
        
        for item in inventory_items:
            status_color = 'red' if '⚠️' in item[8] else 'green'
            self.inventory_tree.insert('', 'end', text=item[0], values=item[1:],
                                      tags=(status_color,))
        
        self.inventory_tree.tag_configure('red', foreground='red')
        self.inventory_tree.tag_configure('green', foreground='green')
        
        # Add scrollbars
        v_scroll = ttk.Scrollbar(inv_frame, orient='vertical', command=self.inventory_tree.yview)
        h_scroll = ttk.Scrollbar(inv_frame, orient='horizontal', command=self.inventory_tree.xview)
        self.inventory_tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        self.inventory_tree.grid(row=0, column=0, sticky='nsew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        h_scroll.grid(row=1, column=0, sticky='ew')
        
        inv_frame.grid_columnconfigure(0, weight=1)
        inv_frame.grid_rowconfigure(0, weight=1)
        
        # Summary panel
        summary_frame = ttk.LabelFrame(inventory, text="Inventory Summary")
        summary_frame.grid(row=2, column=0, sticky='ew', padx=20, pady=10)
        
        summary_labels = [
            ("Total Items:", "234"),
            ("Total Value:", "$12,456.78"),
            ("Low Stock Items:", "8"),
            ("Items to Order:", "12"),
            ("Last Count:", "11/15/2024"),
            ("Next Count Due:", "11/22/2024")
        ]
        
        for i, (label, value) in enumerate(summary_labels):
            tk.Label(summary_frame, text=label, font=("Helvetica", 10, "bold")).grid(row=0, column=i*2, padx=10, pady=5, sticky='e')
            tk.Label(summary_frame, text=value, font=("Helvetica", 10)).grid(row=0, column=i*2+1, padx=(0, 20), pady=5, sticky='w')
    
    def create_invoices_tab(self):
        """Create the invoice processing tab"""
        invoices = self.tabs['invoices']
        invoices.grid_columnconfigure(0, weight=1)
        invoices.grid_rowconfigure(1, weight=1)
        
        # Toolbar
        toolbar = ttk.Frame(invoices)
        toolbar.grid(row=0, column=0, sticky='ew', padx=20, pady=10)
        
        ttk.Button(toolbar, text="📸 Scan Invoice", command=self.scan_invoice,
                  style='Success.TButton').pack(side='left', padx=5)
        ttk.Button(toolbar, text="📁 Import PDF", command=self.import_invoice,
                  style='Primary.TButton').pack(side='left', padx=5)
        ttk.Button(toolbar, text="✏️ Manual Entry", command=self.manual_invoice).pack(side='left', padx=5)
        ttk.Button(toolbar, text="📊 Compare Vendors", command=self.compare_vendors).pack(side='left', padx=5)
        ttk.Button(toolbar, text="💰 Process Payment", command=self.process_payment).pack(side='left', padx=5)
        
        # Date filter
        date_frame = ttk.Frame(toolbar)
        date_frame.pack(side='right', padx=5)
        ttk.Label(date_frame, text="Date Range:").pack(side='left', padx=5)
        self.invoice_date = ttk.Combobox(date_frame, width=20,
                                         values=['This Week', 'This Month', 'Last Month', 'This Year'])
        self.invoice_date.set('This Month')
        self.invoice_date.pack(side='left')
        self.invoice_date.bind('<<ComboboxSelected>>', self.on_invoice_date_change)
        
        # Invoice list
        inv_frame = ttk.Frame(invoices)
        inv_frame.grid(row=1, column=0, sticky='nsew', padx=20, pady=10)
        
        columns = ('Date', 'Invoice #', 'Amount', 'Items', 'Status', 'Due Date', 'Notes')
        self.invoice_tree = ttk.Treeview(inv_frame, columns=columns, show='tree headings')
        
        self.invoice_tree.heading('#0', text='Vendor')
        for col in columns:
            self.invoice_tree.heading(col, text=col)
        
        self.invoice_tree.column('#0', width=150)
        self.invoice_tree.column('Date', width=100)
        self.invoice_tree.column('Invoice #', width=120)
        self.invoice_tree.column('Amount', width=100)
        self.invoice_tree.column('Items', width=80)
        self.invoice_tree.column('Status', width=100)
        self.invoice_tree.column('Due Date', width=100)
        self.invoice_tree.column('Notes', width=200)
        
        # Sample invoices
        invoice_data = [
            ("Shamrock Foods", "11/18/2024", "INV-2024-3456", "$3,245.67", "45", "✅ Processed", "11/25/2024", "Weekly produce order"),
            ("SYSCO", "11/17/2024", "SYS-789012", "$2,156.43", "38", "⏳ Pending", "11/24/2024", "Dry goods restock"),
            ("Local Farm Co", "11/16/2024", "LF-2024-089", "$456.78", "12", "✅ Paid", "11/23/2024", "Organic vegetables"),
            ("Shamrock Foods", "11/15/2024", "INV-2024-3455", "$1,876.54", "31", "✅ Paid", "11/22/2024", "Protein order"),
            ("US Foods", "11/14/2024", "USF-445566", "$987.65", "22", "⚠️ Review", "11/21/2024", "Price discrepancy"),
            ("SYSCO", "11/10/2024", "SYS-789011", "$2,543.21", "41", "✅ Paid", "11/17/2024", "Monthly staples")
        ]
        
        for invoice in invoice_data:
            self.invoice_tree.insert('', 'end', text=invoice[0], values=invoice[1:])
        
        # Scrollbars
        v_scroll = ttk.Scrollbar(inv_frame, orient='vertical', command=self.invoice_tree.yview)
        self.invoice_tree.configure(yscrollcommand=v_scroll.set)
        
        self.invoice_tree.grid(row=0, column=0, sticky='nsew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        
        inv_frame.grid_columnconfigure(0, weight=1)
        inv_frame.grid_rowconfigure(0, weight=1)
        
        # Summary statistics
        summary_frame = ttk.LabelFrame(invoices, text="Invoice Summary")
        summary_frame.grid(row=2, column=0, sticky='ew', padx=20, pady=10)
        
        stats = [
            ("Total This Month:", "$12,456.78"),
            ("Pending Approval:", "3"),
            ("Average Invoice:", "$1,867.45"),
            ("Vendor Count:", "6"),
            ("Savings Identified:", "$521.43"),
            ("Payment Due:", "$4,234.56")
        ]
        
        for i, (label, value) in enumerate(stats):
            tk.Label(summary_frame, text=label, font=("Helvetica", 10, "bold")).grid(row=0, column=i*2, padx=10, pady=5)
            tk.Label(summary_frame, text=value, font=("Helvetica", 10)).grid(row=0, column=i*2+1, padx=(0, 20), pady=5)
    
    def create_catering_tab(self):
        """Create the catering management tab"""
        catering = self.tabs['catering']
        catering.grid_columnconfigure(0, weight=1)
        catering.grid_rowconfigure(1, weight=1)
        
        # Toolbar
        toolbar = ttk.Frame(catering)
        toolbar.grid(row=0, column=0, sticky='ew', padx=20, pady=10)
        
        ttk.Button(toolbar, text="➕ New Event", command=self.new_catering_event,
                  style='Success.TButton').pack(side='left', padx=5)
        ttk.Button(toolbar, text="📝 Edit Event", command=self.edit_catering_event,
                  style='Primary.TButton').pack(side='left', padx=5)
        ttk.Button(toolbar, text="📋 Create BEO", command=self.create_beo).pack(side='left', padx=5)
        ttk.Button(toolbar, text="💰 Quote", command=self.create_quote).pack(side='left', padx=5)
        ttk.Button(toolbar, text="📊 Event Report", command=self.event_report).pack(side='left', padx=5)
        
        # Calendar view toggle
        view_frame = ttk.Frame(toolbar)
        view_frame.pack(side='right', padx=5)
        self.catering_view = ttk.Combobox(view_frame, width=15,
                                          values=['List View', 'Calendar View', 'Timeline'])
        self.catering_view.set('List View')
        self.catering_view.pack(side='left')
        self.catering_view.bind('<<ComboboxSelected>>', self.on_catering_view_change)
        
        # Events list
        events_frame = ttk.Frame(catering)
        events_frame.grid(row=1, column=0, sticky='nsew', padx=20, pady=10)
        
        columns = ('Date', 'Time', 'Guest Count', 'Menu', 'Total', 'Status', 'Contact')
        self.catering_tree = ttk.Treeview(events_frame, columns=columns, show='tree headings')
        
        self.catering_tree.heading('#0', text='Event Name')
        for col in columns:
            self.catering_tree.heading(col, text=col)
        
        self.catering_tree.column('#0', width=200)
        self.catering_tree.column('Date', width=100)
        self.catering_tree.column('Time', width=80)
        self.catering_tree.column('Guest Count', width=100)
        self.catering_tree.column('Menu', width=150)
        self.catering_tree.column('Total', width=100)
        self.catering_tree.column('Status', width=100)
        self.catering_tree.column('Contact', width=150)
        
        # Sample events
        events = [
            ("Johnson Wedding", "12/15/2024", "5:00 PM", "150", "Premium BBQ", "$12,500", "✅ Confirmed", "Sarah Johnson"),
            ("Corporate Lunch - TechCo", "11/22/2024", "12:00 PM", "75", "Business Lunch", "$2,250", "✅ Confirmed", "Mike Chen"),
            ("Birthday Party - Smith", "11/25/2024", "6:00 PM", "40", "Classic BBQ", "$1,600", "⏳ Pending", "John Smith"),
            ("Holiday Party - ABC Inc", "12/20/2024", "7:00 PM", "200", "Holiday Special", "$15,000", "✅ Confirmed", "Lisa Brown"),
            ("Graduation - Davis", "12/10/2024", "2:00 PM", "80", "Casual BBQ", "$3,200", "💰 Deposit", "Tom Davis")
        ]
        
        for event in events:
            self.catering_tree.insert('', 'end', text=event[0], values=event[1:])
        
        # Scrollbar
        v_scroll = ttk.Scrollbar(events_frame, orient='vertical', command=self.catering_tree.yview)
        self.catering_tree.configure(yscrollcommand=v_scroll.set)
        
        self.catering_tree.grid(row=0, column=0, sticky='nsew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        
        events_frame.grid_columnconfigure(0, weight=1)
        events_frame.grid_rowconfigure(0, weight=1)
        
        # Catering metrics
        metrics_frame = ttk.LabelFrame(catering, text="Catering Metrics")
        metrics_frame.grid(row=2, column=0, sticky='ew', padx=20, pady=10)
        
        metrics = [
            ("Events This Month:", "8"),
            ("Total Revenue:", "$28,000"),
            ("Avg Event Size:", "94 guests"),
            ("Conversion Rate:", "73%"),
            ("Upcoming Events:", "5"),
            ("YTD Revenue:", "$245,000")
        ]
        
        for i, (label, value) in enumerate(metrics):
            tk.Label(metrics_frame, text=label, font=("Helvetica", 10, "bold")).grid(row=0, column=i*2, padx=10, pady=5)
            tk.Label(metrics_frame, text=value, font=("Helvetica", 10)).grid(row=0, column=i*2+1, padx=(0, 20), pady=5)
    
    def create_vendors_tab(self):
        """Create the vendor management tab"""
        vendors = self.tabs['vendors']
        vendors.grid_columnconfigure(0, weight=1)
        vendors.grid_rowconfigure(1, weight=1)
        
        # Toolbar
        toolbar = ttk.Frame(vendors)
        toolbar.grid(row=0, column=0, sticky='ew', padx=20, pady=10)
        
        ttk.Button(toolbar, text="➕ Add Vendor", command=self.add_vendor,
                  style='Success.TButton').pack(side='left', padx=5)
        ttk.Button(toolbar, text="📊 Compare Prices", command=self.compare_vendor_prices,
                  style='Primary.TButton').pack(side='left', padx=5)
        ttk.Button(toolbar, text="📈 Performance Report", command=self.vendor_performance).pack(side='left', padx=5)
        ttk.Button(toolbar, text="📞 Contact Info", command=self.vendor_contacts).pack(side='left', padx=5)
        
        # Main content
        content = ttk.Frame(vendors)
        content.grid(row=1, column=0, sticky='nsew', padx=20, pady=10)
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)
        
        # Vendor list
        vendor_frame = ttk.LabelFrame(content, text="Vendor List")
        vendor_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        
        columns = ('Contact', 'Phone', 'Avg Order', 'Last Order', 'Rating')
        self.vendor_tree = ttk.Treeview(vendor_frame, columns=columns, show='tree headings', height=12)
        
        self.vendor_tree.heading('#0', text='Vendor Name')
        for col in columns:
            self.vendor_tree.heading(col, text=col)
        
        self.vendor_tree.column('#0', width=150)
        self.vendor_tree.column('Contact', width=120)
        self.vendor_tree.column('Phone', width=120)
        self.vendor_tree.column('Avg Order', width=100)
        self.vendor_tree.column('Last Order', width=100)
        self.vendor_tree.column('Rating', width=80)
        
        vendor_data = [
            ("Shamrock Foods", "John Miller", "555-0123", "$2,845", "11/18/2024", "⭐⭐⭐⭐⭐"),
            ("SYSCO", "Sarah Lee", "555-0124", "$2,156", "11/17/2024", "⭐⭐⭐⭐"),
            ("US Foods", "Mike Johnson", "555-0125", "$1,234", "11/14/2024", "⭐⭐⭐⭐"),
            ("Local Farm Co", "Tom Green", "555-0126", "$456", "11/16/2024", "⭐⭐⭐⭐⭐"),
            ("Restaurant Depot", "Lisa White", "555-0127", "$876", "11/10/2024", "⭐⭐⭐")
        ]
        
        for vendor in vendor_data:
            self.vendor_tree.insert('', 'end', text=vendor[0], values=vendor[1:])
        
        self.vendor_tree.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Price comparison panel
        compare_frame = ttk.LabelFrame(content, text="Price Comparison Analysis")
        compare_frame.grid(row=0, column=1, sticky='nsew')
        
        compare_text = tk.Text(compare_frame, wrap='word', height=15)
        compare_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        compare_text.insert('1.0', """VENDOR PRICE COMPARISON
Based on analysis of 136 invoices

KEY FINDINGS:
✅ Shamrock Foods: 29.5% lower average pricing
📊 Best categories by vendor:

SHAMROCK FOODS (Best for):
• Proteins: 15% cheaper
• Dairy: 22% cheaper
• Spices: 45% cheaper
• Annual savings potential: $52,000

SYSCO (Best for):
• Dry goods: 8% cheaper
• Paper products: 12% cheaper
• Cleaning supplies: 18% cheaper

RECOMMENDATIONS:
1. Switch spice orders to Shamrock
2. Maintain SYSCO for dry goods
3. Negotiate volume discounts
4. Review quarterly for changes

MONTHLY SAVINGS IDENTIFIED:
Spices: $881/month
Proteins: $1,245/month
Dairy: $456/month
Total: $2,582/month ($30,984/year)""")
        compare_text.config(state='disabled')
    
    def create_costs_tab(self):
        """Create the cost analysis tab"""
        costs = self.tabs['costs']
        costs.grid_columnconfigure(0, weight=1)
        costs.grid_rowconfigure(1, weight=1)
        
        # Analysis options
        toolbar = ttk.Frame(costs)
        toolbar.grid(row=0, column=0, sticky='ew', padx=20, pady=10)
        
        ttk.Label(toolbar, text="Analysis Type:").pack(side='left', padx=5)
        self.analysis_type = ttk.Combobox(toolbar, width=20,
                                     values=['Food Cost', 'Labor Cost', 'Prime Cost', 'Menu Analysis', 'Profit Margins'])
        self.analysis_type.set('Food Cost')
        self.analysis_type.pack(side='left', padx=5)
        self.analysis_type.bind('<<ComboboxSelected>>', self.on_analysis_type_change)
        
        ttk.Button(toolbar, text="🔄 Refresh", command=self.refresh_costs,
                  style='Primary.TButton').pack(side='left', padx=5)
        ttk.Button(toolbar, text="📊 Generate Report", command=self.generate_cost_report).pack(side='left', padx=5)
        
        # Main content area
        content = ttk.Frame(costs)
        content.grid(row=1, column=0, sticky='nsew', padx=20, pady=10)
        
        # Create notebook for different analyses
        cost_notebook = ttk.Notebook(content)
        cost_notebook.pack(fill='both', expand=True)
        
        # Food Cost tab
        food_cost_frame = ttk.Frame(cost_notebook)
        cost_notebook.add(food_cost_frame, text='Food Cost Analysis')
        
        fc_text = tk.Text(food_cost_frame, wrap='word', padx=20, pady=20)
        fc_text.pack(fill='both', expand=True)
        fc_text.insert('1.0', """FOOD COST ANALYSIS - November 2024

OVERALL METRICS:
• Current Food Cost: 31.5%
• Target Food Cost: 30.0%
• Variance: +1.5%
• Monthly Food Purchases: $15,228
• Monthly Food Sales: $48,344

TOP COST ITEMS:
1. Beef Brisket - 38% food cost ($12.45 cost, $32.00 sell)
2. Prime Rib - 42% food cost ($18.75 cost, $44.50 sell)
3. Lobster Tail - 45% food cost ($22.00 cost, $48.00 sell)

OPPORTUNITIES FOR IMPROVEMENT:
✅ Reduce brisket trim waste by 2%
✅ Negotiate better pricing on proteins
✅ Implement portion control on high-cost items
✅ Review and adjust menu prices

CATEGORY BREAKDOWN:
• Proteins: 42% of food cost
• Produce: 18% of food cost
• Dairy: 12% of food cost
• Dry Goods: 15% of food cost
• Beverages: 13% of food cost""")
        fc_text.config(state='disabled')
        
        # Menu Engineering tab
        menu_frame = ttk.Frame(cost_notebook)
        cost_notebook.add(menu_frame, text='Menu Engineering')
        
        menu_text = tk.Text(menu_frame, wrap='word', padx=20, pady=20)
        menu_text.pack(fill='both', expand=True)
        menu_text.insert('1.0', """MENU ENGINEERING MATRIX

STARS (High Profit, High Popularity):
• BBQ Platter - Profit: $18.50, Sales: 145/month
• Smoked Brisket - Profit: $22.00, Sales: 132/month
• Full Rack Ribs - Profit: $16.75, Sales: 98/month

PLOW HORSES (Low Profit, High Popularity):
• Mac & Cheese - Profit: $4.20, Sales: 234/month
• Caesar Salad - Profit: $5.50, Sales: 187/month
• French Fries - Profit: $3.80, Sales: 312/month

PUZZLES (High Profit, Low Popularity):
• Lobster Tail - Profit: $26.00, Sales: 12/month
• Prime Rib - Profit: $25.75, Sales: 18/month
• Surf & Turf - Profit: $32.00, Sales: 8/month

DOGS (Low Profit, Low Popularity):
• Garden Salad - Profit: $3.20, Sales: 23/month
• Veggie Burger - Profit: $4.50, Sales: 15/month
• Fish Tacos - Profit: $5.25, Sales: 19/month

RECOMMENDATIONS:
1. Promote PUZZLES to increase sales
2. Increase prices on PLOW HORSES
3. Consider removing DOGS
4. Maintain STARS visibility""")
        menu_text.config(state='disabled')
        
        # Profit Margin tab
        margin_frame = ttk.Frame(cost_notebook)
        cost_notebook.add(margin_frame, text='Profit Margins')
        
        margin_text = tk.Text(margin_frame, wrap='word', padx=20, pady=20)
        margin_text.pack(fill='both', expand=True)
        margin_text.insert('1.0', """PROFIT MARGIN ANALYSIS

RESTAURANT vs CATERING COMPARISON:

RESTAURANT OPERATIONS:
• Revenue: $20,000/month
• Food Cost: 35%
• Labor Cost: 32%
• Operating Expenses: 29%
• Net Profit Margin: 4%

CATERING OPERATIONS:
• Revenue: $28,000/month
• Food Cost: 28%
• Labor Cost: 18%
• Operating Expenses: 9%
• Net Profit Margin: 45%

KEY INSIGHTS:
✅ Catering is 11x more profitable than restaurant
✅ Lower labor cost % in catering (batch preparation)
✅ Better food cost control in catering
✅ Higher average check in catering

STRATEGIC RECOMMENDATIONS:
1. Focus growth efforts on catering
2. Use restaurant as marketing for catering
3. Optimize restaurant menu for profitability
4. Consider catering-only days
5. Develop corporate catering packages

MONTHLY PROFIT COMPARISON:
Restaurant: $800 (4% of $20,000)
Catering: $12,600 (45% of $28,000)
Total: $13,400 net profit""")
        margin_text.config(state='disabled')
    
    def create_equipment_tab(self):
        """Create the equipment maintenance tab"""
        equipment = self.tabs['equipment']
        equipment.grid_columnconfigure(0, weight=1)
        equipment.grid_rowconfigure(1, weight=1)
        
        # Toolbar
        toolbar = ttk.Frame(equipment)
        toolbar.grid(row=0, column=0, sticky='ew', padx=20, pady=10)
        
        ttk.Button(toolbar, text="➕ Add Equipment", command=self.add_equipment,
                  style='Success.TButton').pack(side='left', padx=5)
        ttk.Button(toolbar, text="🔧 Schedule Maintenance", command=self.schedule_maintenance,
                  style='Primary.TButton').pack(side='left', padx=5)
        ttk.Button(toolbar, text="📝 Log Repair", command=self.log_repair).pack(side='left', padx=5)
        ttk.Button(toolbar, text="📊 Maintenance Report", command=self.maintenance_report).pack(side='left', padx=5)
        
        # Equipment list
        equip_frame = ttk.Frame(equipment)
        equip_frame.grid(row=1, column=0, sticky='nsew', padx=20, pady=10)
        
        columns = ('Model', 'Serial #', 'Last Service', 'Next Service', 'Status', 'Notes')
        self.equipment_tree = ttk.Treeview(equip_frame, columns=columns, show='tree headings')
        
        self.equipment_tree.heading('#0', text='Equipment Name')
        for col in columns:
            self.equipment_tree.heading(col, text=col)
        
        self.equipment_tree.column('#0', width=200)
        self.equipment_tree.column('Model', width=150)
        self.equipment_tree.column('Serial #', width=120)
        self.equipment_tree.column('Last Service', width=100)
        self.equipment_tree.column('Next Service', width=100)
        self.equipment_tree.column('Status', width=100)
        self.equipment_tree.column('Notes', width=200)
        
        equipment_data = [
            ("Convection Oven #1", "Vulcan VC44GD", "VC2024-1123", "10/15/2024", "12/15/2024", "✅ Good", "Regular maintenance"),
            ("Convection Oven #2", "Vulcan VC44GD", "VC2024-1124", "10/15/2024", "12/15/2024", "⚠️ Service Due", "Needs cleaning"),
            ("6-Burner Range", "Wolf C60SS", "WF2023-4567", "09/20/2024", "12/20/2024", "✅ Good", "Working well"),
            ("Walk-in Cooler", "Kolpak P7-0810", "KP2022-8901", "11/01/2024", "12/01/2024", "⚠️ Check Soon", "Temperature fluctuation"),
            ("Dish Machine", "Hobart AM15", "HB2023-3456", "10/30/2024", "11/30/2024", "✅ Good", "New pump installed"),
            ("Smoker", "Ole Hickory EL-SSE", "OH2021-2345", "11/10/2024", "01/10/2025", "✅ Excellent", "Deep cleaned"),
            ("Fryer #1", "Frymaster MJ45", "FM2023-6789", "10/25/2024", "12/25/2024", "✅ Good", "Oil changed weekly"),
            ("Ice Machine", "Manitowoc IY-0606A", "MT2022-4321", "11/05/2024", "12/05/2024", "⚠️ Service Due", "Reduced output")
        ]
        
        for item in equipment_data:
            self.equipment_tree.insert('', 'end', text=item[0], values=item[1:])
        
        # Scrollbar
        v_scroll = ttk.Scrollbar(equip_frame, orient='vertical', command=self.equipment_tree.yview)
        self.equipment_tree.configure(yscrollcommand=v_scroll.set)
        
        self.equipment_tree.grid(row=0, column=0, sticky='nsew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        
        equip_frame.grid_columnconfigure(0, weight=1)
        equip_frame.grid_rowconfigure(0, weight=1)
        
        # Maintenance schedule
        schedule_frame = ttk.LabelFrame(equipment, text="Upcoming Maintenance")
        schedule_frame.grid(row=2, column=0, sticky='ew', padx=20, pady=10)
        
        schedule_text = tk.Text(schedule_frame, height=4, wrap='word')
        schedule_text.pack(fill='both', expand=True, padx=10, pady=10)
        schedule_text.insert('1.0', """• 11/30 - Dish Machine quarterly service
• 12/01 - Walk-in Cooler temperature check
• 12/05 - Ice Machine cleaning and service
• 12/15 - Convection Ovens semi-annual maintenance""")
        schedule_text.config(state='disabled')
    
    def create_orders_tab(self):
        """Create the order guide tab"""
        orders = self.tabs['orders']
        orders.grid_columnconfigure(0, weight=1)
        orders.grid_rowconfigure(1, weight=1)
        
        # Toolbar
        toolbar = ttk.Frame(orders)
        toolbar.grid(row=0, column=0, sticky='ew', padx=20, pady=10)
        
        ttk.Button(toolbar, text="➕ Create Order", command=self.create_order,
                  style='Success.TButton').pack(side='left', padx=5)
        ttk.Button(toolbar, text="🔄 Auto-Generate", command=self.auto_generate_order,
                  style='Primary.TButton').pack(side='left', padx=5)
        ttk.Button(toolbar, text="📧 Send to Vendor", command=self.send_order).pack(side='left', padx=5)
        ttk.Button(toolbar, text="📋 Order History", command=self.order_history).pack(side='left', padx=5)
        
        # Vendor selection
        vendor_frame = ttk.Frame(toolbar)
        vendor_frame.pack(side='right', padx=5)
        ttk.Label(vendor_frame, text="Vendor:").pack(side='left', padx=5)
        self.vendor_select = ttk.Combobox(vendor_frame, width=20,
                                     values=['Shamrock Foods', 'SYSCO', 'US Foods', 'Local Farm Co'])
        self.vendor_select.set('Shamrock Foods')
        self.vendor_select.pack(side='left')
        self.vendor_select.bind('<<ComboboxSelected>>', self.on_vendor_select_change)
        
        # Order guide
        order_frame = ttk.Frame(orders)
        order_frame.grid(row=1, column=0, sticky='nsew', padx=20, pady=10)
        
        columns = ('Category', 'Unit', 'Par', 'On Hand', 'Order Qty', 'Unit Cost', 'Total')
        self.order_tree = ttk.Treeview(order_frame, columns=columns, show='tree headings')
        
        self.order_tree.heading('#0', text='Item Name')
        for col in columns:
            self.order_tree.heading(col, text=col)
        
        self.order_tree.column('#0', width=200)
        self.order_tree.column('Category', width=100)
        self.order_tree.column('Unit', width=80)
        self.order_tree.column('Par', width=60)
        self.order_tree.column('On Hand', width=80)
        self.order_tree.column('Order Qty', width=80)
        self.order_tree.column('Unit Cost', width=80)
        self.order_tree.column('Total', width=100)
        
        # Sample order items
        order_items = [
            ("Beef Brisket", "Proteins", "lb", "20", "8", "12", "$8.00", "$96.00"),
            ("Pork Shoulder", "Proteins", "lb", "40", "45", "0", "$3.50", "$0.00"),
            ("Romaine Lettuce", "Produce", "head", "20", "12", "8", "$2.00", "$16.00"),
            ("Heavy Cream", "Dairy", "qt", "12", "8", "4", "$4.00", "$16.00"),
            ("BBQ Sauce", "Condiments", "gal", "10", "6", "4", "$12.00", "$48.00"),
            ("All Purpose Flour", "Dry Goods", "lb", "40", "50", "0", "$0.75", "$0.00")
        ]
        
        for item in order_items:
            self.order_tree.insert('', 'end', text=item[0], values=item[1:])
        
        # Scrollbar
        v_scroll = ttk.Scrollbar(order_frame, orient='vertical', command=self.order_tree.yview)
        self.order_tree.configure(yscrollcommand=v_scroll.set)
        
        self.order_tree.grid(row=0, column=0, sticky='nsew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        
        order_frame.grid_columnconfigure(0, weight=1)
        order_frame.grid_rowconfigure(0, weight=1)
        
        # Order summary
        summary_frame = ttk.LabelFrame(orders, text="Order Summary")
        summary_frame.grid(row=2, column=0, sticky='ew', padx=20, pady=10)
        
        summary_info = [
            ("Items to Order:", "4"),
            ("Order Total:", "$176.00"),
            ("Estimated Delivery:", "11/20/2024"),
            ("Payment Terms:", "Net 30"),
            ("Order Status:", "Draft")
        ]
        
        for i, (label, value) in enumerate(summary_info):
            tk.Label(summary_frame, text=label, font=("Helvetica", 10, "bold")).grid(row=0, column=i*2, padx=10, pady=5)
            tk.Label(summary_frame, text=value, font=("Helvetica", 10)).grid(row=0, column=i*2+1, padx=(0, 20), pady=5)
    
    def create_reports_tab(self):
        """Create the reports tab"""
        reports = self.tabs['reports']
        reports.grid_columnconfigure(0, weight=1)
        reports.grid_rowconfigure(1, weight=1)
        
        # Report selection
        toolbar = ttk.Frame(reports)
        toolbar.grid(row=0, column=0, sticky='ew', padx=20, pady=10)
        
        ttk.Label(toolbar, text="Report Type:").pack(side='left', padx=5)
        self.report_type = ttk.Combobox(toolbar, width=25,
                                   values=['Daily Sales Summary', 'Weekly P&L', 'Monthly Analysis',
                                          'Vendor Comparison', 'Catering Summary', 'Inventory Valuation'])
        self.report_type.set('Monthly Analysis')
        self.report_type.pack(side='left', padx=5)
        self.report_type.bind('<<ComboboxSelected>>', self.on_report_type_change)
        
        ttk.Button(toolbar, text="🔄 Generate", command=self.generate_report,
                  style='Primary.TButton').pack(side='left', padx=5)
        ttk.Button(toolbar, text="📧 Email", command=self.email_report).pack(side='left', padx=5)
        ttk.Button(toolbar, text="🖨️ Print", command=self.print_report).pack(side='left', padx=5)
        ttk.Button(toolbar, text="💾 Export", command=self.export_report).pack(side='left', padx=5)
        
        # Report display
        report_frame = ttk.Frame(reports)
        report_frame.grid(row=1, column=0, sticky='nsew', padx=20, pady=10)
        
        report_text = tk.Text(report_frame, wrap='word', padx=20, pady=20)
        report_text.pack(fill='both', expand=True)
        
        # Sample monthly report
        report_text.insert('1.0', """THE LARIAT RESTAURANT
MONTHLY OPERATIONS REPORT
November 2024
═══════════════════════════════════════════════════════════════

EXECUTIVE SUMMARY
─────────────────────────────────────────────────────────────
Total Revenue: $48,000
• Restaurant: $20,000 (42%)
• Catering: $28,000 (58%)

Net Profit: $13,400 (27.9% margin)
• Restaurant: $800 (4% margin)
• Catering: $12,600 (45% margin)

REVENUE ANALYSIS
─────────────────────────────────────────────────────────────
Restaurant Operations:
• Average Daily Sales: $667
• Customer Count: 1,850
• Average Check: $10.81
• Top Selling Items:
  1. BBQ Platter - 145 orders
  2. Pulled Pork Sandwich - 132 orders
  3. Brisket Plate - 98 orders

Catering Operations:
• Events Completed: 8
• Average Event Size: 94 guests
• Average Event Revenue: $3,500
• Largest Event: Johnson Wedding (150 guests, $12,500)

COST ANALYSIS
─────────────────────────────────────────────────────────────
Food Cost:
• Restaurant: 35% ($7,000)
• Catering: 28% ($7,840)
• Combined: 30.9% ($14,840)

Labor Cost:
• Restaurant: 32% ($6,400)
• Catering: 18% ($5,040)
• Combined: 23.8% ($11,440)

Prime Cost: 54.7% ($26,280)

VENDOR PERFORMANCE
─────────────────────────────────────────────────────────────
Total Purchases: $14,840
• Shamrock Foods: $8,234 (55.5%)
• SYSCO: $4,123 (27.8%)
• Local Farm Co: $1,456 (9.8%)
• Other: $1,027 (6.9%)

Cost Savings Identified: $2,582
• Vendor optimization opportunity
• 29.5% savings available on spices

OPERATIONAL HIGHLIGHTS
─────────────────────────────────────────────────────────────
✅ Achieved 27.9% net profit margin
✅ Catering revenue up 15% YoY
✅ Food cost controlled at 30.9%
⚠️ Restaurant margin needs improvement
⚠️ Equipment maintenance backlog

KEY RECOMMENDATIONS
─────────────────────────────────────────────────────────────
1. Focus growth on catering (45% margin vs 4%)
2. Implement vendor optimization plan ($30,984/year savings)
3. Review restaurant menu pricing
4. Address equipment maintenance schedule
5. Develop corporate catering packages

UPCOMING PRIORITIES
─────────────────────────────────────────────────────────────
• Johnson Wedding (12/15) - 150 guests
• Holiday catering bookings
• Year-end inventory count
• Q1 2025 menu revision
• Vendor contract negotiations

═══════════════════════════════════════════════════════════════
Report Generated: November 18, 2024
Prepared by: Sean Burdges, Owner/Operator""")
        
        report_text.config(state='disabled')
        
        # Add scrollbar
        scroll = ttk.Scrollbar(report_frame, orient='vertical', command=report_text.yview)
        report_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
    
    def create_statusbar(self):
        """Create status bar at bottom of window"""
        statusbar = ttk.Frame(self.root)
        statusbar.grid(row=1, column=0, sticky='ew')
        
        self.status_text = tk.StringVar()
        self.status_text.set("Ready")
        
        status_label = ttk.Label(statusbar, textvariable=self.status_text, relief='sunken', anchor='w')
        status_label.pack(side='left', fill='x', expand=True, padx=2, pady=2)
        
        # Add connection status
        connection_label = ttk.Label(statusbar, text="✅ Connected", relief='sunken', anchor='e')
        connection_label.pack(side='right', padx=2, pady=2)
    
    def load_initial_data(self):
        """Load initial data from files"""
        self.status_text.set("Loading data...")
        # In a real app, this would load from database or files
        self.status_text.set("Ready")
    
    # Placeholder methods for button commands
    def new_recipe(self):
        messagebox.showinfo("New Recipe", "Opening new recipe dialog...")
    
    def import_data(self):
        filename = filedialog.askopenfilename(title="Import Data",
                                              filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")])
        if filename:
            messagebox.showinfo("Import", f"Importing from {filename}")
    
    def export_data(self):
        filename = filedialog.asksaveasfilename(title="Export Data",
                                               defaultextension=".xlsx",
                                               filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")])
        if filename:
            messagebox.showinfo("Export", f"Exporting to {filename}")
    
    def open_settings(self):
        messagebox.showinfo("Settings", "Opening settings dialog...")
    
    def generate_daily_report(self):
        messagebox.showinfo("Report", "Generating daily sales report...")
    
    def generate_monthly_report(self):
        messagebox.showinfo("Report", "Generating monthly P&L report...")
    
    def generate_vendor_report(self):
        messagebox.showinfo("Report", "Generating vendor analysis report...")
    
    def generate_catering_report(self):
        messagebox.showinfo("Report", "Generating catering summary report...")
    
    def show_documentation(self):
        messagebox.showinfo("Help", "Opening documentation...")
    
    def show_about(self):
        messagebox.showinfo("About", "The Lariat Bible\nVersion 1.0.0\n\nComprehensive Restaurant Management System\n\n© 2024 The Lariat Restaurant")
    
    def edit_recipe(self):
        messagebox.showinfo("Edit", "Opening recipe editor...")
    
    def delete_recipe(self):
        if messagebox.askyesno("Delete", "Are you sure you want to delete this recipe?"):
            messagebox.showinfo("Delete", "Recipe deleted")
    
    def copy_recipe(self):
        messagebox.showinfo("Copy", "Recipe copied")
    
    def calculate_recipe_cost(self):
        messagebox.showinfo("Cost", "Calculating recipe cost...")
    
    def print_recipe(self):
        messagebox.showinfo("Print", "Printing recipe...")
    
    def process_invoice(self):
        messagebox.showinfo("Invoice", "Processing invoice...")
    
    def count_inventory(self):
        messagebox.showinfo("Inventory", "Starting inventory count...")
    
    def create_order(self):
        messagebox.showinfo("Order", "Creating new order...")
    
    def book_catering(self):
        messagebox.showinfo("Catering", "Booking new catering event...")
    
    def generate_report(self):
        messagebox.showinfo("Report", "Generating report...")
    
    def add_inventory(self):
        messagebox.showinfo("Inventory", "Adding inventory item...")
    
    def receive_inventory(self):
        messagebox.showinfo("Receive", "Recording inventory receipt...")
    
    def set_par_levels(self):
        messagebox.showinfo("Par Levels", "Setting par levels...")
    
    def inventory_usage_report(self):
        messagebox.showinfo("Report", "Generating usage report...")
    
    def scan_invoice(self):
        messagebox.showinfo("Scan", "Opening camera for invoice scan...")
    
    def import_invoice(self):
        messagebox.showinfo("Import", "Importing invoice PDF...")
    
    def manual_invoice(self):
        messagebox.showinfo("Manual", "Opening manual invoice entry...")
    
    def compare_vendors(self):
        messagebox.showinfo("Compare", "Comparing vendor prices...")
    
    def process_payment(self):
        messagebox.showinfo("Payment", "Processing payment...")
    
    def new_catering_event(self):
        messagebox.showinfo("Catering", "Creating new catering event...")
    
    def edit_catering_event(self):
        messagebox.showinfo("Edit", "Editing catering event...")
    
    def create_beo(self):
        messagebox.showinfo("BEO", "Creating Banquet Event Order...")
    
    def create_quote(self):
        messagebox.showinfo("Quote", "Creating catering quote...")
    
    def event_report(self):
        messagebox.showinfo("Report", "Generating event report...")
    
    def add_vendor(self):
        messagebox.showinfo("Vendor", "Adding new vendor...")
    
    def compare_vendor_prices(self):
        messagebox.showinfo("Compare", "Comparing vendor prices...")
    
    def vendor_performance(self):
        messagebox.showinfo("Report", "Generating vendor performance report...")
    
    def vendor_contacts(self):
        messagebox.showinfo("Contacts", "Showing vendor contacts...")
    
    def refresh_costs(self):
        messagebox.showinfo("Refresh", "Refreshing cost data...")
    
    def generate_cost_report(self):
        messagebox.showinfo("Report", "Generating cost analysis report...")
    
    def add_equipment(self):
        messagebox.showinfo("Equipment", "Adding new equipment...")
    
    def schedule_maintenance(self):
        messagebox.showinfo("Maintenance", "Scheduling maintenance...")
    
    def log_repair(self):
        messagebox.showinfo("Repair", "Logging equipment repair...")
    
    def maintenance_report(self):
        messagebox.showinfo("Report", "Generating maintenance report...")
    
    def auto_generate_order(self):
        messagebox.showinfo("Auto", "Auto-generating order based on par levels...")
    
    def send_order(self):
        messagebox.showinfo("Send", "Sending order to vendor...")
    
    def order_history(self):
        messagebox.showinfo("History", "Showing order history...")
    
    def email_report(self):
        messagebox.showinfo("Email", "Emailing report...")
    
    def print_report(self):
        messagebox.showinfo("Print", "Printing report...")
    
    def export_report(self):
        messagebox.showinfo("Export", "Exporting report...")

    # Toggle/Combobox Event Handlers
    def on_inventory_filter_change(self, event=None):
        """Handle inventory filter selection changes"""
        filter_value = self.inventory_filter.get()
        messagebox.showinfo("Filter Changed", f"Filtering inventory by: {filter_value}")
        # TODO: Implement actual filtering logic here

    def on_invoice_date_change(self, event=None):
        """Handle invoice date range selection changes"""
        date_range = self.invoice_date.get()
        messagebox.showinfo("Date Range Changed", f"Showing invoices for: {date_range}")
        # TODO: Implement actual date filtering logic here

    def on_catering_view_change(self, event=None):
        """Handle catering view toggle changes"""
        view_type = self.catering_view.get()
        messagebox.showinfo("View Changed", f"Switching to: {view_type}")
        # TODO: Implement view switching logic (List/Calendar/Timeline)

    def on_analysis_type_change(self, event=None):
        """Handle cost analysis type selection changes"""
        analysis = self.analysis_type.get()
        messagebox.showinfo("Analysis Type Changed", f"Analyzing: {analysis}")
        # TODO: Implement analysis type switching logic

    def on_vendor_select_change(self, event=None):
        """Handle vendor selection changes"""
        vendor = self.vendor_select.get()
        messagebox.showinfo("Vendor Selected", f"Showing orders for: {vendor}")
        # TODO: Implement vendor filtering logic

    def on_report_type_change(self, event=None):
        """Handle report type selection changes"""
        report = self.report_type.get()
        messagebox.showinfo("Report Type Changed", f"Report selected: {report}")
        # TODO: Implement report type switching logic

if __name__ == "__main__":
    root = tk.Tk()
    app = LariatBibleApp(root)
    root.mainloop()
