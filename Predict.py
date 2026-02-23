import pandas as pd
import numpy as np
import tensorflow as tf
import pickle, time
import os, datetime, re
from math import pi
from fractions import Fraction
from NN_Structure import Neural_Network_Arch

class Prediction_Package:
    def __init__(self):
        self.date = datetime.date.today()
        self.PATH = os.path.join(os.path.dirname(__file__), "models")

        if not os.path.isdir(self.PATH):
            raise ValueError(f"Invalid path: {self.PATH}")

        self.model_path =  None
        self.WEB_SOURCE = None
        self.scaler_path =  None
        self.encoder_path =  None
        self.target_scaler_path =  None
        self.EXP_FOLDER = None 
        self.S_Box = pd.read_excel("Box_Choice.xlsx", sheet_name="S_Box")
        self.N_Box = pd.read_excel("Box_Choice.xlsx", sheet_name="N_Box")
        self.ALL_Box = pd.read_excel("Box_Choice.xlsx", sheet_name="ALL")
        self.Result = None
        self.LOCATE_PARAMETER()
        self.scaler, self.encoder, self.target_scaler = self.load_preprocessors(
            self.scaler_path, self.encoder_path, self.target_scaler_path
        )
    def LOCATE_PARAMETER(self):
        for doc in os.listdir(self.PATH):
            lower_doc = doc.lower()

            if "target_scaler" in lower_doc:  # typo retained for consistency
                self.target_scaler_path = os.path.join(self.PATH, doc)
            elif "scaler" in lower_doc and "target" not in lower_doc:
                self.scaler_path = os.path.join(self.PATH, doc)
            elif "encoder" in lower_doc:
                self.encoder_path = os.path.join(self.PATH, doc)
            elif "ver" in lower_doc:
                self.model_path = os.path.join(self.PATH, doc)

        if self.encoder_path is None:
            raise ValueError("Can't find encoder")
        if self.scaler_path is None:
            raise ValueError("Can't find scaler")
        if self.target_scaler_path is None:
            raise ValueError("Can't find target_scaler")
        if self.model_path is None:
            raise ValueError("Can't find model")

    def parse_diameter(self, val):
        try:
            if pd.isna(val):
                return None
            val = str(val)
            if 'M' in val:
                return float(val.lstrip('M'))  # M0050 → 5.0 mm
            elif '#' in val:
                return round((int(val.split('#')[1].lstrip('0')) * 0.013 + 0.06) * 25.4, 3)  # I#006 → #6 → 3.5052 mm
            else:
                return None
        except Exception as e:
            print(f"Error parsing diameter {val}: {e}")
            return None

    def preprocess_length(self, val, unit='mm'):
        try:
            if pd.isna(val) or val == "":
                return None
            val = str(val)
            if unit == 'mm':
                return float(val)  # 40 → 40.0 mm
            elif unit == 'inch':
                if "-" in val and "/" in val:
                    inches_len = int(val.split("-")[0]) * 25.4
                    min_inches_len = float(Fraction(val.split("-")[1])) * 25.4
                elif "/" in val:
                    inches_len = 0
                    min_inches_len = float(Fraction(val)) * 25.4
                else:
                    inches_len = int(val) * 25.4
                    min_inches_len = 0
                return inches_len + min_inches_len  # 1-1/4 → 1.25 inch → 31.75 mm
        except Exception as e:
            print(f"Error preprocessing length {val}: {e}")
            return None

    def load_preprocessors(self, scaler_path, encoder_path, target_scaler_path):
        scaler, encoder, target_scaler = None, None, None
        try:
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                print(f"Loaded scaler from {scaler_path}")
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    encoder = pickle.load(f)
                print(f"Loaded encoder from {encoder_path}")
            if os.path.exists(target_scaler_path):
                with open(target_scaler_path, 'rb') as f:
                    target_scaler = pickle.load(f)
                print(f"Loaded target scaler from {target_scaler_path}")
            return scaler, encoder, target_scaler
        except Exception as e:
            print(f"Error loading preprocessors: {str(e)}")
            return None, None, None

    def estimate_screw_volume(self, diam, length):
        if pd.isna(diam) or pd.isna(length) or diam is None or length is None:
            return None
        try:
            radius = diam / 2
            volume = pi * (radius ** 2) * length
            return round(volume, 2)
        except Exception as e:
            print(f"Error calculating screw volume for diam={diam}, length={length}: {e}")
            return None

    def predict_product(self):
        NN_MAC = Neural_Network_Arch(load_erp_data = False)

        try:
            # Load and preprocess input data
            if  self.WEB_SOURCE is None: 
                Predict_Sample = pd.read_excel(os.path.join(self.PATH, "Sample.xlsx"))
            else : 
                # retreive the data from the website
                Predict_Sample = self.WEB_SOURCE

            # Validate required columns
            required_columns = ['Diameter', 'Length', 'Quantity', 'pdc_4', 'pdc_5', 'Screw_Type', 'Head_Type']
            missing_cols = [col for col in required_columns if col not in Predict_Sample.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            Predict_Sample['Length'] = Predict_Sample.apply(
                lambda row: self.preprocess_length(
                    str(row['Length']), unit='inch' if '#' in str(row['Diameter']) else 'mm'
                ), axis=1
            )

            Predict_Sample["Diameter"] = Predict_Sample["Diameter"].apply(self.parse_diameter)
            print(Predict_Sample)
            Predict_Sample["Screw volume in the box"] = Predict_Sample.apply(
                lambda row: self.estimate_screw_volume(row['Diameter'], row['Length']) * row["Quantity"], axis=1
            )
            
            # Check for invalid values
            if Predict_Sample[['Screw volume in the box', 'Diameter', 'Length']].isna().any().any():
                print("Warning: NaN values in numeric features after preprocessing")
                
            if Predict_Sample[['pdc_4', 'pdc_5', 'Screw_Type', 'Head_Type']].isna().any().any():
                print("Warning: NaN values in categorical features after preprocessing")
            

            # Load preprocessors
            if self.scaler is None :
                print("check the scaler path")
            elif  self.encoder is None : 
                print("check the encoder path")
            elif self.target_scaler is None:
                print("check the target_scaler path")
                
            NN_MAC.scaler = self.scaler
            NN_MAC.encoder = self.encoder

            # Prepare features
            NN_MAC.Train_data = Predict_Sample[
                ['Screw volume in the box', 'Diameter', 'Length', 'pdc_4', 'pdc_5', 'Screw_Type', 'Head_Type']
            ]
            X, _, _, _, _ = NN_MAC._prepare_feature_NN(save_files=False)
            #print(X)
            #print(f"Feature shape for prediction: {X.shape}")

            # Load model
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            model = tf.keras.models.load_model(self.model_path)
            print(f"Loaded model: {self.model_path}")

            # Make prediction
            pred = model.predict(X, batch_size = 64)
            # Inverse-transform predictions
            if self.target_scaler is not None:
                pred = self.target_scaler.inverse_transform(pred).flatten()
            else:
                print("no target scaler")
            pred = pred.flatten()
            Predict_Sample['predicted_packing_ratio'] = pred
            Predict_Sample["predicted_decision_volume"] = Predict_Sample["Screw volume in the box"]/Predict_Sample["predicted_packing_ratio"]
            self.Result = Predict_Sample.copy()
            #print(f"target_scaler after prediction: {self.target_scaler}")
            #print(Predict_Sample)
            
        except Exception as e:
            print(f"Error in predict_product: {e}")
            raise
    # prepare data for the energy chart
    def extract_box_volumes(self, row):
        matched = str(row["Matched_Box"])
        screw_vol = row["predicted_decision_volume"]
        print(matched)

        # Initialize
        s_vol_min, s_vol_max, n_vol_min, n_vol_max = [None] * 4

        # Handle S_Box side
        try:
            s_match = matched.split("|")[0]
            s_codes = s_match.split("/") if "/" in s_match else [s_match]
            s_vol_min = self.S_Box[self.S_Box["Quote_Code"] == s_codes[0]]["Volume(mm³)"].values[0]
            s_vol_max = self.S_Box[self.S_Box["Quote_Code"] == s_codes[-1]]["Volume(mm³)"].values[0]
        except:
            pass

        # Handle N_Box side
        try:
            n_match = matched.split("|")[1] if "|" in matched else ""
            n_codes = n_match.split("/") if "/" in n_match else [n_match]
            n_vol_min = self.N_Box[self.N_Box["Quote_Code"] == n_codes[0]]["Volume(mm³)"].values[0]
            n_vol_max = self.N_Box[self.N_Box["Quote_Code"] == n_codes[-1]]["Volume(mm³)"].values[0]
        except:
            pass

        s_text = f"[{s_codes[0]}] {s_vol_min:,} | [Screw] {int(screw_vol):,} | [{s_codes[-1]}] {s_vol_max:,}" if s_vol_min and s_vol_max else ""
        n_text = f"[{n_codes[0]}] {n_vol_min:,} | [Screw] {int(screw_vol):,} | [{n_codes[-1]}] {n_vol_max:,}" if n_vol_min and n_vol_max else ""

        return f"{s_text}\n{n_text}" if s_text or n_text else None


    def Match_Box(self):

        if self.Result is None or self.Result.empty:
            raise ValueError("No prediction results available. Run predict_product first.")

        try:
            
            self.S_Box["Volume(mm³)"] = np.prod(self.S_Box["Volume"].str.split("x", expand=True).astype(int).values, axis=1)
            self.N_Box["Volume(mm³)"] = np.prod(self.N_Box["Volume"].str.split("x", expand=True).astype(int).values, axis=1)
            self.ALL_Box["Volume(mm³)"] = np.prod(self.ALL_Box["Volume"].str.split("x", expand=True).astype(int).values, axis=1)

            available_volumes_S = self.S_Box["Volume(mm³)"].values
            available_volumes_N = self.N_Box["Volume(mm³)"].values

            def find_closest_box(vol):
                try:
                    # For S
                    positive_S = [v for v in available_volumes_S if v >= vol]
                    negative_S = [v for v in available_volumes_S if v < vol]


                    closest_positive_S = min(positive_S, key=lambda v: v - vol) if positive_S else None
                    closest_negative_S = max(negative_S, key=lambda v: v - vol) if negative_S else None

                    # For N
                    positive_N = [v for v in available_volumes_N if v >= vol]
                    negative_N = [v for v in available_volumes_N if v < vol]


                    closest_positive_N = min(positive_N, key=lambda v: v - vol) if positive_N else None
                    closest_negative_N = max(negative_N, key=lambda v: v - vol) if negative_N else None

                    # Get matched codes
                    # For S
                    matched_S_codes = []
                    if closest_positive_S is not None:
                        code_pos_S = self.S_Box.loc[self.S_Box["Volume(mm³)"] == closest_positive_S, "Quote_Code"]
                        matched_S_codes.append(code_pos_S.iloc[0] if not code_pos_S.empty else None)
                    else:
                        matched_S_codes.append(None)

                    if closest_negative_S is not None:
                        code_neg_S = self.S_Box.loc[self.S_Box["Volume(mm³)"] == closest_negative_S, "Quote_Code"]
                        matched_S_codes.append(code_neg_S.iloc[0] if not code_neg_S.empty else None)
                    else:
                        matched_S_codes.append(None)

                    # For N
                    matched_N_codes = []
                    if closest_positive_N is not None:
                        code_pos_N = self.N_Box.loc[self.N_Box["Volume(mm³)"] == closest_positive_N, "Quote_Code"]
                        matched_N_codes.append(code_pos_N.iloc[0] if not code_pos_N.empty else None)
                    else:
                        matched_N_codes.append(None)

                    if closest_negative_N is not None:
                        code_neg_N = self.N_Box.loc[self.N_Box["Volume(mm³)"] == closest_negative_N, "Quote_Code"]
                        matched_N_codes.append(code_neg_N.iloc[0] if not code_neg_N.empty else None)
                    else:
                        matched_N_codes.append(None)


                    return f"{matched_S_codes[0]}/{matched_S_codes[1]}/{matched_N_codes[0]}/{matched_N_codes[1]}"

                except Exception as e:
                    print(f"Error matching box for volume {vol}: {e}")
                    return None
            # try to find the acceptable box
            def is_fit(row):
                box_type = str(row["Box type"])
                matched = str(row["Matched_Box"])

                if not box_type or not matched or "/" not in matched:
                    return 0

                box_prefix = box_type[0]
                
                # Extract number safely (e.g., N4A → 4)
                box_num_match = re.match(r"[A-Z](\d+)", box_type)
                if not box_num_match:
                    return 0
                box_num = int(box_num_match.group(1))

                matched_parts = matched.split("/")

                for part in matched_parts:
                    if part and part[0] == box_prefix:
                        match_num = re.match(r"[A-Z](\d+)", part)
                        if match_num:
                            matched_num = int(match_num.group(1))
                            if matched_num >= box_num:
                                return 1
                return 0

            self.Result["Matched_Box"] = self.Result["predicted_decision_volume"].apply(find_closest_box)
            self.Result["Fit_in_Option"] = self.Result.apply(is_fit, axis=1)
            self.Result["Statistic"] = self.Result.apply(lambda row : 1 if str(row["Box type"]) in str(row["Matched_Box"]) else 0, axis = 1)
            self.Result["Volume_Chart"] = self.Result.apply(self.extract_box_volumes, axis=1)
            self.Result.to_excel(os.path.join(self.PATH, f"Test_result_{self.date}.xlsx"))
            print("Box matching completed. Results saved to Test_result_{}.xlsx".format(self.date))
            print("The Accuracy would be below : ")
            print("All item : {}/ : {}".format(self.Result.shape[0], self.Result["Statistic"].sum()))
            print("Precision : {:.2f}%".format(self.Result["Statistic"].sum()*100/self.Result.shape[0]))
            print("Acceptable Choice Rate  : {:.2f}%".format(self.Result["Fit_in_Option"].sum()*100/self.Result.shape[0]))

            print(self.Result)

        except Exception as e:
            print(f"Error in Match_Box: {e}")
            raise

if __name__ == "__main__":
    start = time.time()
    input_data = pd.DataFrame([{
            "Diameter": "#8",
            "Length": "4",
            "Quantity": int(1000),
            "pdc_4": "-",
            "pdc_5": "-",
            "Screw_Type": "SDS",
            "Head_Type": "PAN",
            "Box type" : None
        }])

    bot = Prediction_Package()
    #bot.WEB_SOURCE = input_data
    bot.predict_product()
    bot.Match_Box()
    print("耗時{}".format(time.time()- start))