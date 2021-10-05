import sys
sys.path.append('../')
from scipy.integrate import solve_ivp
import os
import phydrus as ps


class SMModels:
    class HerradaModel:
        @classmethod
        def herrada(self, t, wc, k_sat, ksi_sat, bc_lambda, theta_sat, theta_res, n_layers, total_depth, forcingdata):
            d_z = total_depth / n_layers

            def flux(idx_layer):
                '''
                idx_layer:int, from 1 to the N_layer-2 (originally 2 to N-1)
                '''

                def k(theta):
                    phi = (theta - theta_res) / (theta_sat - theta_res)
                    power = (2 + 3 * bc_lambda) / bc_lambda
                    return k_sat * (phi ** power)

                def ksi(theta):
                    phi = (theta - theta_res) / (theta_sat - theta_res)
                    power = -1 / bc_lambda
                    return ksi_sat * (phi ** power)

                theta_hat = (wc[idx_layer] + wc[idx_layer - 1]) / 2

                return k(theta_hat) * (1 - (ksi(wc[idx_layer]) - ksi(wc[idx_layer - 1])) / d_z)

            d_theta = [0 for _ in range(len(wc))]  # rate of change in water content (theta)
            # print(self.P(self,t))
            # The condition for upper boundary
            # if wc[0] >= theta_sat and sm_models.P(self, t, forcings) >= flux(1):
            print('{:.1f}'.format(t), end='\r')
            e_pot = forcingdata[1][int(round(t))]
            Corr = (wc[0] - theta_res) / (theta_sat - theta_res)
            if (e_pot > 0.0 and Corr > 1e-2):
                if wc[0] > theta_res + flux(1) / d_z:
                    e_t = (wc[0] - flux(1) / d_z - theta_res) / (theta_sat - theta_res) * e_pot
                else:
                    e_t = 0.0
            else:
                e_p = 0.0
                e_t = 0.0
            q_rain = forcingdata[0][int(round(t))]
            if wc[0] >= theta_sat and q_rain >= flux(1):
                # print('HERE')
                wc[0] = theta_sat
                d_theta[n_layers] = q_rain - flux(1)
            else:
                # d_theta[0] = (sm_models.P(self, t, forcings) - flux(1)) / d_z
                d_theta[0] = (q_rain - flux(1) - e_t) / d_z

            for layer in range(1, n_layers - 2, 1):
                d_theta[layer] = (flux(layer) - flux(layer + 1)) / d_z

            wc[n_layers - 1] = (4 * wc[n_layers - 2] - wc[n_layers - 3]) / 3

            return d_theta

        @classmethod
        def run(self, params, t_span, init):
            K_sat, Ksi_sat, Lambda, theta_sat, theta_res, N_layer, T_depth, forcings = params

            results = solve_ivp(
                lambda t, wc: self.herrada(t, wc, K_sat, Ksi_sat, Lambda, theta_sat, theta_res, N_layer,
                                           T_depth, forcings),
                t_span, init, t_eval=range(int(t_span[0]), int(t_span[1]), 1))
            return results

    class hlm_254:
        """
        Order of parameters: A_i L_i A_h invtau  k_2 k_i c_1 c_2
        # The numbering is:    0    1   2   3       4   5   6   7
        The input is as below:
        precip: {'t':[t_start:t_end], 'precip':[p_1,..p_t_end]},
        hs_params: A dictionary with variables as:
        Order of global_params: v_0  lambda_1  lambda_2  v_h k_3 k_I_factor  h_b S_L A  B  exponent  v_B
        The numbering is:       0    1         2         3   4   5           6   7   8  9  10         11
        global_params: A dictionary with variables as:
        Order of parameters: A_i L_i A_h invtau  k_2 k_i c_1 c_2
        # The numbering is:  0    1   2   3       4   5   6   7
        """
        global global_params, forcings
        def __init__(self, hs_params, global_params):
            self.params = self.Precalculations(hs_params)
            self.global_params = global_params



        @classmethod
        def Precalculations(self, hs_params, global_params):
            ''' Order of parameters: A_i L_i A_h invtau  k_2 k_i c_1 c_2
            # The numbering is:    0    1   2   3       4   5   6   7 '''

            # Spatially distributed parameters
            A_i = hs_params[0]  # Upstream Area[m ^ 2]
            L_i = hs_params[1]  # Link length[m]
            A_h = hs_params[2]  # Hillslope Area[m ^ 2]
            # kSat = hs_params[3]  # Saturated hydraulic conductivity[cm / hr]
            # BC_lambda = hs_params[4]  # Brooks - Corey's lambda [-]
            # theta_s = hs_params[5]  # Saturated water content[-]
            # theta_r = hs_params[6]  # Residual water content[-]
            # psi_Sat = hs_params[7]  # Bubbling pressure [log10(kPa)]

            # Global Parameters (AKA spatially constant parameters)
            v_0 = global_params[0]  # Channel Routing: Channel reference velocity [m/min]
            lambda_1 = global_params[1]  # Channel Routing: Power law exponent 1 [-]
            lambda_2 = global_params[2]  # Channel Routing: Power law exponent 2 [-]
            v_h = global_params[3]  # Hillslope velocity [m/min]
            k_i_factor = global_params[5]  #

            # Update the parameters
            hs_params.insert(3, 60.0 * v_0 * pow(A_i, lambda_2) / ((1.0 - lambda_1) * L_i))  # [1 / min] invtau
            hs_params.insert(4, v_h * L_i / A_h * 60.0)  # k_2 [1 / min]
            hs_params.insert(5, hs_params[4] * k_i_factor)  # k_i [1 / min]
            hs_params.insert(6, 0.001 / 60.0)  # c_1 (mm / hr->m / min)
            hs_params.insert(7, A_h / 60.0)  # c_2

            # Update the hs_params array
            # hs_params[8] = theta_s
            # hs_params[9] = theta_r
            # hs_params[10] = BC_lambda
            # hs_params[11] = -1.0 * pow(10.0, psi_Sat) * 0.10197  # psi_sat [log10 (kPa)]     -> mH2O
            # hs_params[12] = pow(10.0, kSat) * 0.01 / 60.0  # K_sat [log10 (cm / hr)] -> [m / min]
            return hs_params

        @classmethod
        def TopLayerHillslope(self, t, y_i, global_params, params, forcings):
            # lambda_1 = global_params[1]
            k_3 = global_params[4]  # [1 / min]
            h_b = global_params[6]  # [m]
            S_L = global_params[7]  # [m]
            A = global_params[8]
            B = global_params[9]
            exponent = global_params[10]
            # v_B = global_params[11]
            e_pot = forcings[1][int(round(t))] * (1e-3 / (30.0 * 24.0 * 60.0))  # [mm / month] -> [m / min]
            L = params[1]  # [m]
            A_h = params[2]  # [m ^ 2]
            h_r = params[3]  # [m]
            invtau = params[3]  # [1 / min]
            k_2 = params[4]  # [1 / min]
            k_i = params[5]  # [1 / min]
            c_1 = params[6]
            c_2 = params[7]
            # states from previous time step (initial states)
            q = y_i[0]  # [m ^ 3 / s]
            s_p = y_i[1]  # [m]
            s_t = y_i[2]  # [m]
            s_s = y_i[3]  # [m]
            s_precip = y_i[4]  # [m]
            V_r = y_i[5]  # [m ^ 3]
            q_b = y_i[6]  # [m ^ 3 / s]
            print('{:.1f}'.format(t), end='\r')
            # Evaporation
            Corr = s_p + s_t / S_L + s_s / (h_b - S_L)
            if (e_pot > 0.0 and Corr > 1e-12):
                e_p = s_p * e_pot / Corr
                e_t = s_t / S_L * e_pot / Corr
                e_s = s_s / (h_b - S_L) * e_pot / Corr
            else:
                e_p = 0.0
                e_t = 0.0
                e_s = 0.0

            if (1.0 - s_t / S_L > 0.0):
                pow_term = pow(1.0 - s_t / S_L, exponent)
            else:
                pow_term = 0.0
            k_t = (A + B * pow_term) * k_2

            # Fluxes
            q_pl = k_2 * s_p  # [m / min]
            q_pt = k_t * s_p  # [m / min]
            q_ts = k_i * s_t  # [m / min]
            q_sl = k_3 * s_s  # [m / min]

            ans = [0 for _ in range(7)]
            # Discharge
            ans[0] = -q + (q_pl + q_sl) * c_2
            # for i in range(0, num_parents):
            #     ans[0] += y_p[i * dim]
            # ans[0] = invtau * pow(q, lambda_1) * ans[0]

            # Hillslope
            ans[1] = forcings[0][int(round(t))] * c_1 - q_pl - q_pt - e_p
            ans[2] = q_pt - q_ts - e_t
            ans[3] = q_ts - q_sl - e_s

            # Additional states
            ans[4] = forcings[0][int(round(t))] * c_1
            ans[5] = q_pl
            ans[6] = q_sl * A_h
            # for i in range(0, num_parents):
            #     ans[6] += y_p[i * dim + 6] * 60.0
            # ans[6] *= v_B / L
            return ans

        @classmethod
        def run(self, glob_params_254, hs_params, t_span, init, forcings):
            # glob_params_254 = [0.33, 0.2, -0.1, 0.1, 2.0425e-6, 0.02, 1.0, 0.05, 0.0, 99, 3.0, 0.75]
            precalc_254 = self.Precalculations(hs_params, glob_params_254)
            res = solve_ivp(lambda t, ans: self.TopLayerHillslope(t, ans, glob_params_254, precalc_254,
                                                                  forcings),
                            t_span, init, t_eval=range(int(t_span[0]), int(t_span[1]), 60))
            return res

    class hlm_254_et_hr:
        """
        Order of parameters: A_i L_i A_h invtau  k_2 k_i c_1 c_2
        # The numbering is:    0    1   2   3       4   5   6   7
        The input is as below:
        precip: {'t':[t_start:t_end], 'precip':[p_1,..p_t_end]},
        hs_params: A dictionary with variables as:
        Order of global_params: v_0  lambda_1  lambda_2  v_h k_3 k_I_factor  h_b S_L A  B  exponent  v_B
        The numbering is:       0    1         2         3   4   5           6   7   8  9  10         11
        global_params: A dictionary with variables as:
        Order of parameters: A_i L_i A_h invtau  k_2 k_i c_1 c_2
        # The numbering is:  0    1   2   3       4   5   6   7
        """
        global global_params, forcings
        def __init__(self, hs_params, global_params):
            self.params = self.Precalculations(hs_params)
            self.global_params = global_params



        @classmethod
        def Precalculations(self, hs_params, global_params):
            ''' Order of parameters: A_i L_i A_h invtau  k_2 k_i c_1 c_2
            # The numbering is:    0    1   2   3       4   5   6   7 '''

            # Spatially distributed parameters
            A_i = hs_params[0]  # Upstream Area[m ^ 2]
            L_i = hs_params[1]  # Link length[m]
            A_h = hs_params[2]  # Hillslope Area[m ^ 2]
            # kSat = hs_params[3]  # Saturated hydraulic conductivity[cm / hr]
            # BC_lambda = hs_params[4]  # Brooks - Corey's lambda [-]
            # theta_s = hs_params[5]  # Saturated water content[-]
            # theta_r = hs_params[6]  # Residual water content[-]
            # psi_Sat = hs_params[7]  # Bubbling pressure [log10(kPa)]

            # Global Parameters (AKA spatially constant parameters)
            v_0 = global_params[0]  # Channel Routing: Channel reference velocity [m/min]
            lambda_1 = global_params[1]  # Channel Routing: Power law exponent 1 [-]
            lambda_2 = global_params[2]  # Channel Routing: Power law exponent 2 [-]
            v_h = global_params[3]  # Hillslope velocity [m/min]
            k_i_factor = global_params[5]  #

            # Update the parameters
            hs_params.insert(3, 60.0 * v_0 * pow(A_i, lambda_2) / ((1.0 - lambda_1) * L_i))  # [1 / min] invtau
            hs_params.insert(4, v_h * L_i / A_h * 60.0)  # k_2 [1 / min]
            hs_params.insert(5, hs_params[4] * k_i_factor)  # k_i [1 / min]
            hs_params.insert(6, 0.001 / 60.0)  # c_1 (mm / hr->m / min)
            hs_params.insert(7, A_h / 60.0)  # c_2

            # Update the hs_params array
            # hs_params[8] = theta_s
            # hs_params[9] = theta_r
            # hs_params[10] = BC_lambda
            # hs_params[11] = -1.0 * pow(10.0, psi_Sat) * 0.10197  # psi_sat [log10 (kPa)]     -> mH2O
            # hs_params[12] = pow(10.0, kSat) * 0.01 / 60.0  # K_sat [log10 (cm / hr)] -> [m / min]
            return hs_params

        @classmethod
        def TopLayerHillslope(self, t, y_i, global_params, params, forcings):
            # lambda_1 = global_params[1]
            k_3 = global_params[4]  # [1 / min]
            h_b = global_params[6]  # [m]
            S_L = global_params[7]  # [m]
            A = global_params[8]
            B = global_params[9]
            exponent = global_params[10]
            # v_B = global_params[11]
            e_pot = forcings[1][int(round(t))] * (1e-3 / ( 60.0))  # [mm / hr] -> [m / min]
            L = params[1]  # [m]
            A_h = params[2]  # [m ^ 2]
            h_r = params[3]  # [m]
            invtau = params[3]  # [1 / min]
            k_2 = params[4]  # [1 / min]
            k_i = params[5]  # [1 / min]
            c_1 = params[6]
            c_2 = params[7]
            # states from previous time step (initial states)
            q = y_i[0]  # [m ^ 3 / s]
            s_p = y_i[1]  # [m]
            s_t = y_i[2]  # [m]
            s_s = y_i[3]  # [m]
            s_precip = y_i[4]  # [m]
            V_r = y_i[5]  # [m ^ 3]
            q_b = y_i[6]  # [m ^ 3 / s]
            print('{:.1f}'.format(t), end='\r')
            # Evaporation
            Corr = s_p + s_t / S_L + s_s / (h_b - S_L)
            if (e_pot > 0.0 and Corr > 1e-12):
                e_p = s_p * e_pot / Corr
                e_t = s_t / S_L * e_pot / Corr
                e_s = s_s / (h_b - S_L) * e_pot / Corr
            else:
                e_p = 0.0
                e_t = 0.0
                e_s = 0.0

            if (1.0 - s_t / S_L > 0.0):
                pow_term = pow(1.0 - s_t / S_L, exponent)
            else:
                pow_term = 0.0
            k_t = (A + B * pow_term) * k_2

            # Fluxes
            q_pl = k_2 * s_p  # [m / min]
            q_pt = k_t * s_p  # [m / min]
            q_ts = k_i * s_t  # [m / min]
            q_sl = k_3 * s_s  # [m / min]

            ans = [0 for _ in range(7)]
            # Discharge
            ans[0] = -q + (q_pl + q_sl) * c_2
            # for i in range(0, num_parents):
            #     ans[0] += y_p[i * dim]
            # ans[0] = invtau * pow(q, lambda_1) * ans[0]

            # Hillslope
            ans[1] = forcings[0][int(round(t))] * c_1 - q_pl - q_pt - e_p
            ans[2] = q_pt - q_ts - e_t
            ans[3] = q_ts - q_sl - e_s

            # Additional states
            ans[4] = forcings[0][int(round(t))] * c_1
            ans[5] = q_pl
            ans[6] = q_sl * A_h
            # for i in range(0, num_parents):
            #     ans[6] += y_p[i * dim + 6] * 60.0
            # ans[6] *= v_B / L
            return ans

        @classmethod
        def run(self, glob_params_254, hs_params, t_span, init, forcings):
            # glob_params_254 = [0.33, 0.2, -0.1, 0.1, 2.0425e-6, 0.02, 1.0, 0.05, 0.0, 99, 3.0, 0.75]
            precalc_254 = self.Precalculations(hs_params, glob_params_254)
            res = solve_ivp(lambda t, ans: self.TopLayerHillslope(t, ans, glob_params_254, precalc_254,
                                                                  forcings),
                            t_span, init, t_eval=range(int(t_span[0]), int(t_span[1]), 60))
            return res

    class herrada_model_mod:
        global forcings

        @classmethod
        def herrada(self, t, wc, K_sat, Ksi_sat, Lambda, theta_sat, theta_res, N_layer, T_depth, forcings, k_2):
            D_z = T_depth / N_layer

            def flux(idx_layer):
                def K(theta):
                    phi = (theta - theta_res) / (theta_sat - theta_res)
                    power = (2 + 3 * Lambda) / Lambda
                    return K_sat * (phi ** power)

                def Ksi(theta):
                    phi = (theta - theta_res) / (theta_sat - theta_res)
                    power = -1 / Lambda
                    return Ksi_sat * (phi ** power)
                theta_hat = (wc[idx_layer] + wc[idx_layer - 1]) / 2
                return K(theta_hat) * (1 - (Ksi(wc[idx_layer]) - Ksi(wc[idx_layer - 1])) / D_z)

            def flux_new():
                def K(theta):
                    phi = (theta - theta_res) / (theta_sat - theta_res)
                    power = (2 + 3 * Lambda) / Lambda
                    return K_sat * (phi ** power)

                def Ksi(theta):
                    phi = (theta - theta_res) / (theta_sat - theta_res)
                    power = -1 / Lambda
                    return Ksi_sat * (phi ** power)

                theta_hat = (wc[1] + theta_sat) / 2

                return K(theta_hat) * (1 - (Ksi(wc[1]) - Ksi(theta_sat)) / D_z)

            d_theta = [0 for _ in range(len(wc))]  # rate of change in water content (theta)
            # print(self.P(self,t))
            # The condition for upper boundary
            # if wc[0] >= theta_sat and sm_models.P(self, t, forcings) >= flux(1):
            print('{:.1f}'.format(t), end='\r')
            e_pot = forcings[1][int(round(t))]
            Corr = (wc[0] - theta_res) / (theta_sat - theta_res)
            if (e_pot > 0.0 and Corr > 1e-2):
                if wc[0] > theta_res + flux(1) / D_z:
                    e_t = (wc[0] - flux(1) / D_z - theta_res) / (theta_sat - theta_res) * e_pot
                else:
                    e_t = 0.0
            else:
                e_p = 0.0
                e_t = 0.0

            q_rain = forcings[0][int(round(t))]
            q_inf = flux(1)
            if q_rain >= abs(q_inf):
                # print('HERE')
                d_theta[N_layer] = q_rain - flux(1) - wc[N_layer] * k_2 - e_t
                d_theta[0] = (flux_new() - flux(1) - e_t) / D_z
            else:
                # d_theta[0] = (sm_models.P(self, t, forcings) - flux(1)) / D_z
                if wc[N_layer] > 0:
                    d_theta[N_layer] = - wc[N_layer] * k_2
                    d_theta[0] = (q_rain - flux(1)) / D_z
                else:
                    d_theta[0] = (q_rain - flux(1) - e_t) / D_z
                    # d_theta[0] = (q_rain - flux(1)) / D_z

            for layer in range(1, N_layer - 2, 1):
                d_theta[layer] = (flux(layer) - flux(layer + 1)) / D_z

            wc[N_layer - 1] = (4 * wc[N_layer - 2] - wc[N_layer - 3]) / 3
            # d_theta[N_layer+1] = wc[0] * 0.0001
            return d_theta

        @classmethod
        def run(self, params, t_span, init):
            K_sat, Ksi_sat, Lambda, theta_sat, theta_res, N_layer, T_depth, forcings, k_2 = params

            results = solve_ivp(
                lambda t, wc: self.herrada(t, wc, K_sat, Ksi_sat, Lambda, theta_sat, theta_res, N_layer,
                                           T_depth, forcings, k_2),
                t_span, init, t_eval=range(int(t_span[0]), int(t_span[1]), 1))
            return results

    # TODO: rewrite @herrada_model_mod with new equation for top layer
 
    class GSSHA:

        @classmethod
        def herrada(self, t, wc, K_sat, Ksi_sat, Lambda, theta_sat, theta_res, N_layer, T_depth, forcings, k_2):
            D_z = [1, 2, 2, 5, 10, 30, 50, 100, 100, 100, 100]
            
            def flux(idx_layer):
                k_scheme = 'else'
                def K(theta):
                    phi = (theta - theta_res) / (theta_sat - theta_res)
                    power = (2 + 3 * Lambda) / Lambda
                    return K_sat * (phi ** power)

                def Ksi(theta):
                    phi = (theta - theta_res) / (theta_sat - theta_res)
                    power = -1 / Lambda
                    return Ksi_sat * (phi ** power)

                depth_total = (D_z[idx_layer] + D_z[idx_layer - 1])
                theta_hat = (D_z[idx_layer] * wc[idx_layer] + D_z[idx_layer - 1] * wc[idx_layer - 1]) / depth_total
                k_hat = (D_z[idx_layer] * K(wc[idx_layer]) + D_z[idx_layer - 1] * K(wc[idx_layer - 1]))/ depth_total
                if k_scheme=='herrada':
                    return K(theta_hat) * (1 - (Ksi(wc[idx_layer]) - Ksi(wc[idx_layer - 1])) / D_z[idx_layer])
                else:
                    return k_hat* (1 - (Ksi(wc[idx_layer]) - Ksi(wc[idx_layer - 1])) / D_z[idx_layer])
            def flux_new():
                k_scheme = 'else'
                def K(theta):
                    phi = (theta - theta_res) / (theta_sat - theta_res)
                    power = (2 + 3 * Lambda) / Lambda
                    return K_sat * (phi ** power)

                def Ksi(theta):
                    phi = (theta - theta_res) / (theta_sat - theta_res)
                    power = -1 / Lambda
                    return Ksi_sat * (phi ** power)

                theta_hat = (wc[0] + theta_sat) / 2
                k_hat = (wc[0] + K(theta_sat)) / 2
                if k_scheme=='herrada':
                    return K(theta_hat) * (1 - (Ksi(wc[0]) - wc[N_layer]) / D_z[0])
                else:
                    return k_hat* (1 - (Ksi(wc[0]) - wc[N_layer]) / D_z[0])
            def flux_new1():
                k_scheme = 'else'
                def K(theta):
                    phi = (theta - theta_res) / (theta_sat - theta_res)
                    power = (2 + 3 * Lambda) / Lambda
                    return K_sat * (phi ** power)

                def Ksi(theta):
                    phi = (theta - theta_res) / (theta_sat - theta_res)
                    power = -1 / Lambda
                    return Ksi_sat * (phi ** power)

                theta_hat = (wc[0] + theta_sat) / 2
                k_hat = (K(wc[0]) + K(theta_sat)) / 2
                if k_scheme=='herrada':
                    return K(theta_hat) * (1 - (Ksi(wc[0]) - Ksi(theta_sat)) / D_z[0])
                else:
                    return k_hat* (1 - (Ksi(wc[0]) - Ksi(theta_sat)) / D_z[0])                
            d_theta = [0 for _ in range(len(wc))]  # rate of change in water content (theta)

            print('{:.1f}'.format(t), end='\r')
            e_pot = forcings[1][int(round(t))]
            Corr = (wc[0] - theta_res) / (theta_sat - theta_res)
            if (e_pot > 0.0 and Corr > 1e-2):
                # if wc[0] > theta_res + flux(1) / D_z[0]:
                e_t =  e_pot
                # else:
                    # e_t = 0.0
            else:
                e_p = 0.0
                e_t = 0.0

            q_rain = forcings[0][int(round(t))]
            q_inf = flux_new1()
            if q_rain >= q_inf:
                # print('HERE')
                # wc[0]=theta_sat
                d_theta[N_layer] = q_rain -flux_new1()- wc[N_layer] * k_2 
                d_theta[0] = (flux_new1()-flux(1) - e_t/5) / D_z[0]
            else:
                # d_theta[0] = (sm_models.P(self, t, forcings) - flux(1)) / D_z
                d_theta[N_layer] = - wc[N_layer] * k_2
                d_theta[0] = (q_rain - flux(1)- e_t/5) / D_z[0]
                    # d_theta[0] = (q_rain - flux(1)) / D_z

            for layer in range(1, N_layer - 2, 1):
                if layer<5:
                    d_theta[layer] = (flux(layer) - flux(layer + 1)-e_t/5)/ D_z[layer]
                else:
                    d_theta[layer] = (flux(layer) - flux(layer + 1))/ D_z[layer]
            # wc[N_layer - 1] = (4 * wc[N_layer - 2] - wc[N_layer - 3]) / 3 
            d_theta[N_layer-1] = (4 * d_theta[N_layer - 2] - d_theta[N_layer - 3]) / 3  - 0.0000154 * wc[N_layer-1]#(4 * d_theta[N_layer - 2] - d_theta[N_layer - 3]) / 3 
            d_theta[N_layer+1] = wc[N_layer] * k_2
            return d_theta

        @classmethod
        def run(cls, params, t_span, init):
            k_sat, ksi_sat, bc_lambda, theta_sat, theta_res, n_layer, t_depth, forcings, k_2 = params

            results = solve_ivp(
                lambda t, wc: cls.herrada(t, wc, k_sat, ksi_sat, bc_lambda, theta_sat, theta_res, n_layer,
                                            t_depth, forcings, k_2),
                t_span, init, t_eval=range(int(t_span[0]), int(t_span[1]), 1))
            return results


    class hlm_10012:

        """
        Order of parameters: A_i L_i A_h invtau  k_2 k_i c_1 c_2
        # The numbering is:    0    1   2   3       4   5   6   7
        The input is as below:
        precip: {'t':[t_start:t_end], 'precip':[p_1,..p_t_end]},
        hs_params: A dictionary with variables as:
        Order of global_params: v_0  lambda_1  lambda_2  v_h k_3 k_I_factor  h_b S_L A  B  exponent  v_B
        The numbering is:       0    1         2         3   4   5           6   7   8  9  10         11
        global_params: A dictionary with variables as:
        Order of parameters: A_i L_i A_h invtau  k_2 k_i c_1 c_2
        # The numbering is:  0    1   2   3       4   5   6   7
        """
        global global_params, forcings
        def __init__(self, hs_params, global_params):
            self.params = self.Precalculations(hs_params)
            self.global_params = global_params



        @classmethod
        def Precalculations(self, hs_params, global_params):
            ''' Order of parameters: A_i L_i A_h invtau  k_2 k_i c_1 c_2
            # The numbering is:    0    1   2   3       4   5   6   7 '''

            # Spatially distributed parameters
            A_i = hs_params[0]              # Upstream Area[m ^ 2]
            L_i = hs_params[1]              # Link length[m]
            A_h = hs_params[2]              # Hillslope Area[m ^ 2]
            kSat = hs_params[3]             # Saturated hydraulic conductivity[cm / hr]
            bc_lambda = hs_params[4]        # Brooks - Corey's lambda [-]
            theta_s = hs_params[5]          # Saturated water content[-]
            theta_r = hs_params[6]          # Residual water content[-]
            psi_Sat = hs_params[7]          # Bubbling pressure [log10(kPa)]

            # Global Parameters (AKA spatially constant parameters)
            v_0 = global_params[0]          # Channel Routing: Channel reference velocity [m/min]
            lambda_1 = global_params[1]     # Channel Routing: Power law exponent 1 [-]
            lambda_2 = global_params[2]     # Channel Routing: Power law exponent 2 [-]
            v_h = global_params[3]          # Hillslope velocity [m/min]
            k_i_factor = global_params[5]  #

            # Update the parameters
            hs_params.insert(3, 60.0 * v_0 * pow(A_i, lambda_2) / ((1.0 - lambda_1) * L_i))  # [1 / min] invtau
            hs_params.insert(4, v_h * L_i / A_h * 60.0)  # k_2 [1 / min]
            hs_params.insert(5, hs_params[4] * k_i_factor)  # k_i [1 / min]
            hs_params.insert(6, 0.001 / 60.0)  # c_1 (mm / hr->m / min)
            hs_params.insert(7, A_h / 60.0)  # c_2

            # Update the hs_params array
            hs_params.insert(8, pow(10.0, kSat) * 0.01 / 60.0)          # K_sat [log10 (cm / hr)] -> [m / min]
            hs_params.insert(9, theta_s)
            hs_params.insert(10, theta_r)
            hs_params.insert(11, bc_lambda)
            hs_params.insert(12, -1.0 * pow(10.0, psi_Sat) * 0.10197)   # psi_sat [log10 (kPa)]     -> mH2O
            return hs_params

        @classmethod
        def hlm_10012(self, t, y_i, global_params, params, forcing_values):
            # lambda_1 = global_params[1]
            def flux_inf_head( theta_1,  theta_2,  K_sat,  psi_sat,  bc_lambda,  theta_s,  theta_r,  S_L1,  S_L2, s_p):
                depth_total = (S_L2 + S_L1)
                phi_1 = (theta_1 - theta_r) / (theta_s - theta_r)
                phi_2 = (theta_2 - theta_r) / (theta_s - theta_r)
                # K
                power_k = (2.0 + 3.0 * bc_lambda) / bc_lambda
                K_layer1 = K_sat * pow(phi_1, power_k)
                K_layer2 = K_sat * pow(phi_2, power_k)
                # Geometric average for K
                K_hat = (S_L1 * K_layer1 + S_L2 * K_layer2) / depth_total

                # psi
                power_psi = -1.0 / bc_lambda
                psi_1 = psi_sat * pow(phi_1, power_psi)
                psi_2 = psi_sat * pow(phi_2, power_psi)
                return K_hat * (1.0 - (psi_2 - psi_1 - s_p) / S_L2)


            def flux_inf_pond( theta_1,  theta_2,  K_sat,  psi_sat,  bc_lambda,  theta_s,  theta_r,  S_L1,  S_L2):
                phi_1 = (theta_s - theta_r) / (theta_s - theta_r)
                phi_2 = (theta_2 - theta_r) / (theta_s - theta_r)
                # K
                power_k = (2.0 + 3.0 * bc_lambda) / bc_lambda
                K_layer1 = K_sat * pow(phi_1, power_k)
                K_layer2 = K_sat * pow(phi_2, power_k)
                # Simple averaging for K
                K_hat = (K_layer1 * S_L1 + K_layer2 * S_L2) / (S_L1 + S_L2)
                # TODO:
                # Add geometric Averaging for K             
                # psi
                power_psi = -1.0 / bc_lambda
                psi_1 = psi_sat * pow(phi_1, power_psi)
                psi_2 = psi_sat * pow(phi_2, power_psi)
                return K_hat * (1.0 - (psi_2 - psi_1) / S_L2)
            
            def flux_inf_pond_GA(theta_1, theta_2, K_sat, psi_sat, bc_lambda, theta_s, theta_r, S_L1, S_L2):
                phi_1 = (theta_s - theta_r) / (theta_s - theta_r)
                phi_2 = (theta_2 - theta_r) / (theta_s - theta_r)
                power_k = (2.0 + 3.0 * bc_lambda) / bc_lambda
                K_layer1 = K_sat * pow(phi_1, power_k)
                K_layer2 = K_sat * pow(phi_2, power_k)
                # Simple averaging for K
                K_hat = (K_layer1 * S_L1 + K_layer2 * S_L2) / (S_L1 + S_L2)
                # TODO: add geometric Averaging for K
                # float K_hat = pow(K_layer1 *K_layer2, 0.5);
                # psi
                power_psi = -1.0 / bc_lambda
                #psi_1 = psi_sat * pow(phi_1, power_psi)
                psi_2 = psi_sat * pow(phi_2, power_psi)
                return K_hat * (1.0 - (psi_2 - S_L1) / (S_L2))

            #lambda_1 =      global_params[1]
            k_3 =           global_params[4] #[1/min]
            #h_b =           global_params[6] #[m]

            #A =             global_params[8]
            #B =             global_params[9]
            #exponent =      global_params[10]
            #v_B =           global_params[11]
            pond_inf_coef = 1 #global_params[12]

            e_pot = forcing_values[1][int(round(t))] * (1e-3 / (30.0 * 24.0 * 60.0)) #[mm/month] -> [m/min]
            
            #L =             params[1]	   #[m]
            A_h =           params[2]	   #[m^2]

            #invtau =        params[3] #[1/min]
            k_2 =           params[4]	   #[1/min]
            #k_i =           params[5]	   #[1/min]
            c_1 =           params[6]
            c_2 =           params[7]
            # Soil parameters
            K_sat =         params[8]
            theta_s =       (int)(params[9] * 1000) / 1000.0
            theta_r =       params[10]
            bc_lambda =     params[11]
            psi_sat =       params[12]
            #L_Top =         5.0
            #h_b =           global_params[6]  # [m]
            # Soil layer depths
            S_L =       [0.01, 0.05, 0.05, 0.10, 0.30, 0.50, 1.0, 1.0, 1.0, 1.0 ] # layer depths [m]

            # states from previous time step (initial states)
            q =         y_i[0]  # [m ^ 3 / s]
            s_p =       y_i[1]  # [m]

            # initialize soil moisture for soil layers
            s_t = [0.0] * 10   # [m] [2-11]
            for i in range(0, 10):
                s_t[i] = y_i[i + 2]
                if (s_t[i] > theta_s):
                    s_t[i] = theta_s
                elif (s_t[i] <= theta_r):
                    s_t[i] = theta_r + 0.01
                
            s_s =       y_i[12]  # [m]
            # s_precip =  y_i[13]  # [m]
            # V_r =       y_i[14]  # [m ^ 3]
            q_b =       y_i[15]  # [m ^ 3 / s]
            q_rain = forcing_values[0][int(round(t))] * c_1
            print('{:.1f}'.format(t), end='\r')
            
            #q_rain = forcing_values[0] * c_1 # // m/min
            q_sl = k_3 * s_t[5] * S_L[5]
            q_pl = k_2 * s_p
            q_t = [0.0] * 9
            for i in range(9):
                q_t[i] = flux_inf_head(s_t[i], s_t[i + 1],
                K_sat, psi_sat, bc_lambda, theta_s, theta_r,
                S_L[i], S_L[i + 1], s_p)

            et_tot = 0.0
            e_pot_rem = e_pot


            # Evaporation
            e_t = [0.0] * 10
            for i in range(10):
                #print(e_pot_rem)
                if (e_pot_rem>1e-11):
                    s_lim = s_t[i] - theta_r - 0.1
                    C_et = 0.5 *( s_lim / pow(0.0001+ pow(s_lim, 2.0), 0.5)) +0.498
                    if (C_et > 0.001):
                        #e_t[i] = s_t[i] - (C_et*e_pot_rem*h)/S_L[i] > theta_r + q_t[i] / S_L[i] + 0.01 if C_et*e_pot_rem else 0.0
                        e_t[i] = C_et*e_pot_rem
                        e_t[i] = e_t[i] > 1e-7 if e_t[i] else 0.0
                        e_pot_rem = e_pot_rem - e_t[i]
                        et_tot += e_t[i]
                else:
                    break
            

            q_inf = q_t[0]
            q_pond_inf = flux_inf_pond(theta_s, s_t[0], K_sat, psi_sat, bc_lambda, theta_s, theta_r, S_L[0], S_L[1])
            
            if (s_p == 0.0):
                if (q_rain > q_pond_inf):
                    dsp = q_rain - q_pond_inf - q_pl
                    ds0 = (q_pond_inf - q_inf - e_t[0]) / S_L[0]
                    #extra_flux = ds0 * h + s_t[0] > theta_s if ds0 - (theta_s - s_t[0]) / h else 0.0
                    #dsp = q_rain - q_pond_inf - q_pl + extra_flux * S_L[0]
                    #ds0 = (q_pond_inf - q_inf - e_t[0]) / S_L[0] - extra_flux
                else:
                    dsp = 0.0
                    ds0 = (q_rain - q_inf - e_t[0]) / S_L[0]
                    #extra_flux = ds0 * h + s_t[0] > theta_s if ds0 - (theta_s - s_t[0]) / h else 0.0
                    #dsp = extra_flux * S_L[0]
                    #ds0 = (q_rain - q_inf - e_t[0]) / S_L[0] - extra_flux
            else:
                q_pond_inf = flux_inf_pond_GA(theta_s, s_t[0], K_sat*pond_inf_coef, psi_sat, bc_lambda, theta_s, theta_r, s_p, S_L[0])
                #q_pond_inf = q_pond_inf > (theta_s - s_t[0]) / h * S_L[0] if (theta_s - s_t[0]) / h * S_L[0] else q_pond_inf
                dsp = q_rain - q_pond_inf - q_pl
                ds0 = (q_pond_inf - q_inf) / S_L[0]
                #extra_flux = ds0 * h + s_t[0] > theta_s if ds0 - (theta_s - s_t[0]) / h else 0.0
                #dsp = q_rain - q_pond_inf - q_pl + extra_flux * S_L[0]
                #ds0 = (q_pond_inf - q_inf) / S_L[0] - extra_flux

            
            ans = [0.0] * 16
            
            # Discharge
            ans[0] = -q + (q_pl + q_sl) * c_2


            # Hillslope
            ans[1] = dsp # S_P
            ans[2] = ds0 # S_t[0]
            
            for i in range(1, 9):
                ans[i + 2] = (q_t[i - 1] - q_t[i] - e_t[i]) / S_L[i] # S_t[1-9]
            
            #ans[7] = (q_t[4] - q_t[5]- e_t[5]) / S_L[5] - k_3 * s_t[5]

            ans[11] = (4.0 * ans[10] - ans[9]) / 3.0 # good SM
            ans[12] = et_tot # S_s
            # Additional states
            ans[13] = forcing_values[0][int(round(t))] * c_1
            ans[14] = q_pl # q_pl
            ans[15] = q_sl * A_h - q_b * 60.0
            return ans

        @classmethod
        def run(self, glob_params_10012, hs_params, t_span, init, forcings):
            # glob_params_254 = [0.33, 0.2, -0.1, 0.1, 2.0425e-6, 0.02, 1.0, 0.05, 0.0, 99, 3.0, 0.75]
            precalc_10012 = self.Precalculations(hs_params, glob_params_10012)
            res = solve_ivp(lambda t, ans: self.hlm_10012(t, ans, glob_params_10012, precalc_10012,
                                                                  forcings),
                            t_span, init, t_eval=range(int(t_span[0]), int(t_span[1]), 60))
            return res

# TODO:
# add new function for separating van-Genuchten model and Brooks-Corey's soil-water characteristics

# TODO:
# Finalize the class for Hydrus1D and test
class hydrus1D:
    @classmethod
    def hydrus_run(cls, t_span, gage_data, vg_params, init_head, bottom, ws="c:\\hydros_", dx=1):
        # TODO: custom obsservation nodes for extracting data from Hydrus1D
        # TODO: add an argument for time interval of the data\

        # reference: https://github.com/phydrus/phydrus

        # Folder where the Hydrus files are to be stored
        # ws = "c:\\output_custom11"
        # exe = os.path.join(os.getcwd(), "../../hydrus")
        exe = os.path.join(os.getcwd(), "F:\\Apps\\phydrus-master\\source\\hydrus.exe")
        # Create model and define units of lenth and concentration
        ml = ps.Model(exe_name=exe, ws_name=ws, name="model1",
                      mass_units="mmol", time_unit="hours", length_unit="cm")
        # Create a HYDRUS1D model object
        ml.add_time_info(tinit=0, tmax=t_span[1])

        ml.add_waterflow(top_bc=3, bot_bc=4, model=3)
        m = ml.get_empty_material_df(n=1)
        m.loc[0:1] = vg_params
        ml.add_material(m)

        # Define loop for potential root water uptake distribution proposed by Hoffman and Van Genuchten
        # bottom = -500  # Depth of the soil column
        #nodes = 100  # Dictretize into n nodes
        # ihead = -10  # Determine initial pressure head

        profile = ps.create_profile(bot=bottom, dx=dx, h=init_head, conc=1e-10, mat=1, lay=1)
        ml.add_profile(profile)
        ml.add_obs_nodes([-5, -10, -20, -50])

        # %%
        # p_1', 'p_2', 'st_5', 'st_10', 'st_20', 'st_50', 'sm_5', 'sm_10',
        # 'sm_20', 'sm_50', 'gwd', 'gwt', 'pet_x', 'pet_y', 'cTop'
        # data as
        # ['tAtm', 'Prec', 'rSoil', 'rRoot', 'hCritA']
        dt_vec = [ts for ts in range(1, t_span[1] + 2)]
        atm1 = gage_data[['p_1', 'pet']]
        atm1 = atm1.reset_index()
        atm1['dt'] = dt_vec
        atm1 = atm1.rename(columns={'dt': 'tAtm', 'p_1': 'Prec', 'pet': 'rSoil'})
        atm1['Prec'] /= 10
        atm1['rSoil'] /= 10

        atm1["cTop"] = 0.0
        atm1.loc[0, "cTop"] = 1.0
        # ml.__delattr__('atmosphere')
        # %%
        ml.add_atmospheric_bc(atm1, hcrits=0)
        # %%
        # Write the input and check if the model simulates
        ml.write_input()
        ml.simulate()
        return ml

    # TODO:
    # rewrite @herrada_model_mod with new equation for top layer
