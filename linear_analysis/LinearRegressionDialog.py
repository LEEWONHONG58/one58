from PyQt5.QtCore import QStringListModel, QVariant, Qt
from PyQt5.QtWidgets import QDialog, QTableWidgetItem, QTableWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QListView, QGridLayout, QPushButton, QMessageBox, QSizePolicy
import numpy as np
from qgis.core import QgsProject, QgsVectorLayer, QgsApplication
from sklearn.linear_model import LinearRegression
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


class LinearRegressionDialog(QDialog):
    """
    LinearRegression class
    Author : WH Lee
    LAB : LSDS
    Date : 2024.07.17 ~ 2024.07.31
    """
    def __init__(self):
        super(LinearRegressionDialog, self).__init__()
        self.init_ui()
        self.setup_connections()
        
    def reset_ui(self):
        self.populateShapefiles()
        self.dependentBox.clear()
        self.dependentBox.addItem("", None)
        self.fieldModel.setStringList([])
        self.independentModel.setStringList([])
    
    def init_ui(self):
        # Create Layout, Widgets
        self.mainLayout = QVBoxLayout()

        # Select Layer Layout
        self.selectLayout = QVBoxLayout()
        self.targetLabel = QLabel("Select a Target Layer:")
        self.selectLayout.addWidget(self.targetLabel)
        self.targetBox = QComboBox()
        self.selectLayout.addWidget(self.targetBox)

        self.dependentLabel = QLabel("Dependent Variable:")
        self.selectLayout.addWidget(self.dependentLabel)
        self.dependentBox = QComboBox()
        self.selectLayout.addWidget(self.dependentBox)

        self.mainLayout.addLayout(self.selectLayout)

        self.fieldLayout = QGridLayout()

        self.fieldLabel = QLabel("Fields")
        self.fieldLayout.addWidget(self.fieldLabel, 0, 0, alignment=Qt.AlignCenter)
        self.fieldView = QListView()
        self.fieldView.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.fieldModel = QStringListModel()
        self.fieldView.setModel(self.fieldModel)
        self.fieldLayout.addWidget(self.fieldView, 1, 0, 2, 1)

        self.inButton = QPushButton(">")
        self.inButton.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.inButton.setFixedSize(50, 20)
        self.fieldLayout.addWidget(self.inButton, 1, 1, alignment=Qt.AlignCenter)
        self.outButton = QPushButton("<")
        self.outButton.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.outButton.setFixedSize(50, 20)
        self.fieldLayout.addWidget(self.outButton, 2, 1, alignment=Qt.AlignCenter)

        self.independentLabel = QLabel("Independent Variables")
        self.fieldLayout.addWidget(self.independentLabel, 0, 2, alignment=Qt.AlignCenter)
        self.independentView = QListView()
        self.independentView.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.independentModel = QStringListModel()
        self.independentView.setModel(self.independentModel)
        self.fieldLayout.addWidget(self.independentView, 1, 2, 2, 1)
        
        self.mainLayout.addLayout(self.fieldLayout)

        self.buttonLayout = QHBoxLayout()
        self.runButton = QPushButton("Run")
        self.cancelButton = QPushButton("Cancel")
        self.buttonLayout.addStretch()
        self.buttonLayout.addWidget(self.runButton)
        self.buttonLayout.addWidget(self.cancelButton)
        self.runButton.setFixedSize(self.cancelButton.sizeHint())

        self.mainLayout.addLayout(self.buttonLayout)

        self.setLayout(self.mainLayout)
        self.setGeometry(300, 200, 350, 300)

    def setup_connections(self):
        self.targetBox.currentTextChanged.connect(self.updateFieldsAndDependentBox)
        self.inButton.clicked.connect(self.addIndependentVariable)
        self.outButton.clicked.connect(self.removeIndependentValueable)
        self.runButton.clicked.connect(self.runRegression)
        self.cancelButton.clicked.connect(self.close)
        
    def populateShapefiles(self):
        self.targetBox.clear()  # ComboBox 초기화
        layers = QgsProject.instance().mapLayers().values()
        # 벡터 레이어 중에서 Shapefile 형식만 필터링
        valid_layers = [layer for layer in layers if isinstance(layer, QgsVectorLayer) and layer.dataProvider().storageType() == 'ESRI Shapefile']
        if valid_layers:
            self.targetBox.addItem("", None)  # 첫 선택지 추가
            for layer in valid_layers:
                self.targetBox.addItem(layer.name(), layer)
        else:
            self.targetBox.addItem("No shapefile layers available", None)

    def updateFieldsAndDependentBox(self):
        self.dependentBox.clear()
        self.dependentBox.addItem("", None)
        self.fieldModel.setStringList([])
        self.independentModel.setStringList([])
        current_layer = self.targetBox.currentData()  # 현재 선택된 레이어 데이터 가져오기
        if current_layer:
            numeric_fields = []
            for field in current_layer.fields():
                # 모든 피처를 검사하여 해당 필드가 숫자형 데이터만 포함하는지 확인
                all_values_numeric = True
                for feature in current_layer.getFeatures():
                    value = feature[field.name()]
                    if not isinstance(value, (int, float, type(None))):  # None은 결측치 허용
                        all_values_numeric = False
                        break

                if all_values_numeric:  # 모든 값이 숫자형인 필드만 추가
                    numeric_fields.append(field.name())

            if numeric_fields:
                self.fieldModel.setStringList(numeric_fields)
                self.dependentBox.addItems(numeric_fields)
            else:
                self.fieldModel.setStringList(["No numeric fields"])
                self.dependentBox.addItem("No numeric fields available")

    def addIndependentVariable(self):
        selected_indexes = self.fieldView.selectedIndexes()
        if selected_indexes:
            selected_field = selected_indexes[0].data()
            field_list = self.fieldModel.stringList()
            independent_fields = self.independentModel.stringList()
            if selected_field not in independent_fields:
                independent_fields.append(selected_field)
                field_list.remove(selected_field)
                self.independentModel.setStringList(independent_fields)
                self.fieldModel.setStringList(field_list)
    
    def removeIndependentValueable(self):
        selected_indexes = self.independentView.selectedIndexes()
        if selected_indexes:
            selected_field = selected_indexes[0].data()
            field_list = self.fieldModel.stringList()
            independent_fields = self.independentModel.stringList()
            if selected_field not in field_list:
                field_list.append(selected_field)
                independent_fields.remove(selected_field)
                self.fieldModel.setStringList(field_list)
                self.independentModel.setStringList(independent_fields)
                
    def runRegression(self):
        current_layer = self.targetBox.currentData()
        if current_layer is None:
            QMessageBox.warning(self, "Error", "Target Layer를 선택하세요")
            return
        dependent_field = self.dependentBox.currentText()
        independent_fields = self.independentModel.stringList()
        if not dependent_field or not independent_fields:
            QMessageBox.warning(self, "Error", "독립변수, 종속변수가 올바르게 선택되었는지 확인하세요")
            return

        y = []
        X = []
        for feature in current_layer.getFeatures():
            y_value = feature[dependent_field]
            if y_value is None:
                continue
            y.append(y_value)
            x_row = []
            for field in independent_fields:
                x_value = feature[field]
                if x_value is None:
                    x_value = 0
                x_row.append(x_value)
            X.append(x_row)
        
        if not y or not X:
            QMessageBox.warning(self, "Error", "독립변수, 종속변수 데이터가 올바른 형식이 아닙니다.")
            return

        X = np.array(X)
        y = np.array(y)

        model = LinearRegression()
        model.fit(X, y)

        # intercept, slope, R^2, adj_R^2
        intercept = model.intercept_
        coefficients = model.coef_
        r_squared = model.score(X, y)
        n = len(y)
        k = len(independent_fields)
        adj_r_squared = 1 - ((1- r_squared) * (n - 1) / (n - k - 1))
        
        # SSres -> standard estimation & RMSE & t_stats & p_values
        predictions = model.predict(X)
        residuals = y - predictions
        SSres = np.sum((residuals[:, np.newaxis] * X) ** 2, axis=0)
        Msres = SSres / (n - k - 1)
        std = Msres
        mse = np.mean((y - predictions) ** 2)
        rmse = np.sqrt(mse * np.linalg.inv(np.dot(X.T, X)).diagonal())
        t_stats = coefficients / rmse
        p_values = [2 * (1 - stats.t.cdf(np.abs(t), df=n-len(coefficients)-1)) for t in t_stats]
        
        # Intercept statistics
        X_with_intercept = np.column_stack((np.ones(n), X))
        intercept_std = np.sqrt(mse * np.linalg.inv(np.dot(X_with_intercept.T, X_with_intercept))[0, 0])
        mse_intercept = np.mean((y - model.predict(X)) ** 2)
        intercept_rmse = np.sqrt(mse_intercept * np.linalg.inv(np.dot(X_with_intercept.T, X_with_intercept))[0, 0])
        intercept_t_stat = intercept / intercept_rmse
        intercept_p_value = 2 * (1 - stats.t.cdf(np.abs(intercept_t_stat), df=n - k - 1))
        intercept_stats = [intercept, intercept_std, intercept_t_stat, intercept_p_value]
        
        vif = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        
        self.showResults(independent_fields, intercept_stats, coefficients, r_squared, adj_r_squared, std, t_stats, p_values, vif)

    def showResults(self, independent_fields, intercept_stats, coefficients, r_squared, adj_r_squared, std, t_stats, p_values, vif):
        resultDialog = RegressResults(independent_fields, intercept_stats, coefficients, r_squared, adj_r_squared, std, t_stats, p_values, vif)
        resultDialog.exec_()
        
    def show(self):
        self.reset_ui()
        super().show()
        self.raise_()
        self.activateWindow()

class RegressResults(QDialog):
    """
    RegressResults class
    Author : WH Lee
    LAB : LSDS
    Date : 2024.07.23 ~ 2024.07.31
    """
    def __init__(self, independent_fields, intercept_stats, coefficients, r_squared, adj_r_squared, std, t_stats, p_values, vif):
        super(RegressResults, self).__init__()
        self.independent_fields = independent_fields
        self.intercept_stats = intercept_stats
        self.coefficients = coefficients
        self.std = std
        self.t_stats = t_stats
        self.p_values = p_values
        self.r_squared = r_squared
        self.adj_r_squared = adj_r_squared
        self.vif = vif
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        # R-squared and Adjusted R-squared labels
        stats_layout = QHBoxLayout()
        r2_label = QLabel(f"R²: {self.r_squared:.3f}")
        adj_r2_label = QLabel(f"Adjusted R²: {self.adj_r_squared:.3f}")
        stats_layout.addWidget(r2_label)
        stats_layout.addWidget(adj_r2_label)
        layout.addLayout(stats_layout)
        # Table for the regression results
        self.table = QTableWidget(self)
        self.table.setColumnCount(6)
        self.table.setRowCount(len(self.coefficients) + 1)
        self.table.setHorizontalHeaderLabels(["Variable", "Coefficients", "Standard Error", "t stats", "P Value", "VIF"])
        
        # intercept Stats
        self.table.setItem(0, 0, QTableWidgetItem("intercept"))
        self.table.setItem(0, 1, QTableWidgetItem(f"{self.intercept_stats[0]:.4f}"))
        self.table.setItem(0, 2, QTableWidgetItem(f"{self.intercept_stats[1]:.4f}"))
        self.table.setItem(0, 3, QTableWidgetItem(f"{self.intercept_stats[2]:.2f}"))
        self.table.setItem(0, 4, QTableWidgetItem(f"{self.intercept_stats[3]:.4f}" if self.intercept_stats[3] > 0.0001 else "<.0001"))
        self.table.setItem(0, 5, QTableWidgetItem(""))
        
        # Fill in the rows for each results
        for i, coef in enumerate(self.coefficients):
            self.table.setItem(i + 1, 0, QTableWidgetItem(self.independent_fields[i]))
            self.table.setItem(i + 1, 1, QTableWidgetItem(f"{self.coefficients[i]:.4f}"))
            self.table.setItem(i + 1, 2, QTableWidgetItem(f"{self.std[i]:.4f}"))
            self.table.setItem(i + 1, 3, QTableWidgetItem(f"{self.t_stats[i]:.2f}"))
            self.table.setItem(i + 1, 4, QTableWidgetItem(f"{self.p_values[i]:.4f}" if self.p_values[i] > 0.0001 else "<.0001"))
            self.table.setItem(i + 1, 5, QTableWidgetItem(f"{self.vif[i]:.4f}"))
        
        layout.addWidget(self.table)
        self.setLayout(layout)
        self.setWindowTitle("Regression Results")
        self.setGeometry(660, 225, 675, 300)