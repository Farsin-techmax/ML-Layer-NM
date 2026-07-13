def process_service_data(mastersheet: str,servhistory: str, rfm: str, appointshow: str, appointnoshow: str, appointdf: str, digidf: str,vhc: str,servcode: str,filter_date: str,last_service_code: int) -> pd.DataFrame:
    """
    Process service history data from CSV file.
    
    Args:
        mastersheet (str): Path to the master sheet CSV file
        servhistory (str): Path to the service history CSV file
        filter_date (str): Date string in YYYY-MM-DD format
        last_service_code (int): Code for the last service
        
    Returns:
        pd.DataFrame: Processed and filtered service data
    """
    logger.info("Starting service data processing")
    
    try:
        # Convert filter date
        filterdate = pd.to_datetime(filter_date)
        logger.info(f'Filter date: {filterdate}')
        
        # Validate input files
        # if not os.path.exists(mastersheet):
        #     logger.error(f"Master sheet not found: {mastersheet}")
        #     raise FileNotFoundError(f"Master sheet not found: {mastersheet}")
        
        if not os.path.exists(servhistory):
            logger.error(f"Service history file not found: {servhistory}")
            raise FileNotFoundError(f"Service history file not found: {servhistory}")
            
        # Read and process master sheet
        
        try:
            
            logger.info("Reading master sheet data")
            col_name = f"Expected{last_service_code}kDate"
            # mastertrain = pd.read_csv(mastersheet, low_memory=False)
            mastertrain = pd.read_parquet('/refactored_test_dir/refactored_mastersheet.parquet')
            logger.debug(f"Master sheet initial shape: {mastertrain.shape}")
            
            
            logger.info("Converting dates in master sheet")
            mastertrain['Last Service Date - PMS'] = pd.to_datetime(mastertrain['Last Service Date - PMS'],format='mixed',dayfirst=True,errors='coerce')
            mastertrain[col_name] = pd.to_datetime(mastertrain[col_name],format='mixed',dayfirst=True)
            mastertrain = mastertrain.query(f"{col_name} <= @filterdate and `Last Service Date - PMS` <= @filterdate")
            logger.info(f"Master sheet shape after date filtering: {mastertrain.shape}")
            servcode_desc = pd.read_csv(servcode)
            servcode_desc = servcode_desc.rename(columns={'SO_CO_CODE': 'Service_Code'})
            # mastertrain.to_csv('validatecode/mastertrain_initial.csv', index=False)
        except Exception as e:
            logger.error(f"Error processing master sheet: {str(e)}")
            raise
        
        # Read and process service history
        try:
            logger.info("Reading service history data")
            serv1 = pd.read_csv(servhistory,low_memory=False,encoding='ISO-8859-1') #Input Service History
            logger.debug(f"Service history initial shape: {serv1.shape}")
            
            logger.info("Processing service history data")
            serv1['Service_Date'] = pd.to_datetime(serv1['Service_Date'],format='mixed',dayfirst=True,errors='coerce')
            invalid_dates = serv1['Service_Date'].isna().sum()
            if invalid_dates > 0:
                logger.warning(f"Found {invalid_dates} invalid dates in service history. Dropping them.")
                serv1 = serv1.dropna(subset=['Service_Date'])
            
            serv1 = serv1.query(f"Service_Date <= @filterdate")
            logger.info(f"Service history shape after date filtering: {serv1.shape}")
            
            logger.info("Converting numeric columns")
            serv1['Mileage'] = pd.to_numeric(serv1['Mileage'],errors='coerce')
            serv1['Revenue'] = pd.to_numeric(serv1['Revenue'],errors='coerce')
            
            logger.info("Processing service numbers")
            serv1['Service_Num'] = serv1['Description'].apply(extract_k)
            serv1['Service_Num'] = np.where(serv1['Description'] == '<=10', 10, serv1['Service_Num'])
            
            # Filter and process service data
            logger.info("Filtering and processing service data")
            serv1 = serv1.query(f"Service_Num <= {last_service_code}")
            serv = serv1[serv1['Vin_No'].isin(mastertrain['VIN'].unique())]
            servM = serv
            serv.to_csv('validatecode/serv_filteredpmsapp.csv', index=False)
            print("Completed filtering service data for PMS features")
            serv = serv.query("Description != 'Others'")
        
            # serv.to_csv('validatecode/serv_filteredpmschkmileage.csv', index=False)
            
            logger.info(f'Max service number in data: {serv["Service_Num"].max()}')
            logger.info(f'Max Service date in data: {serv["Service_Date"].max()}')
        except Exception as e:
            logger.error(f"Error processing service history: {str(e)}")
            raise
        
        # Aggregate service data
        logger.info("Aggregating service data")
        newserv20k = serv.groupby(['Vin_No']).agg(
            Service_Date=('Service_Date', 'max'),
            Mileage=('Mileage', 'max'),
            Total_Revenue=('Revenue', 'sum'),
        ).reset_index()
        
        logger.info(f'UNIQUE VINS in service data: {newserv20k["Vin_No"].nunique()}')

        # Feature Derivation
        logger.info("Starting feature derivation")
        
        # Feature derivation starts here
        try:
            logger.info("Deriving PMS features")
            dfpmsdate = derive_pms_features1(serv1, last_service_code)

            logger.info("Last Non PMS Mileage calculation")
            lastnonpmsmil = get_last_nonpms_mileage(serv1,mastertrain, vin_col="Vin_No", mileage_col="Mileage",
                            date_col="Service_Date")
            
            logger.info("Deriving NPMS features")
            dfnpmsdate = derive_npms_features(serv1)
            
            logger.info("Deriving NonPMS Mileage")
            res1 = get_last_nonpms_before_targetpms(serv1, "Vin_No","Mileage","Service_Date",mastertrain)
            res2 = get_last_nonpms_mileage(serv1, mastertrain,vin_col="Vin_No", mileage_col="Mileage",date_col="Service_Date")
            lastnonpmsmil = pd.concat([res1,res2],axis=0)

            logger.info("Calculating average mileage intervals for PMS")
            avg_mileage_interval = derive_pms_mileage_features(serv1, dfpmsdate, last_service_code)
            
            logger.info("Calculating average mileage intervals for NPMS")
            avg_mileage_interval_non_pms = derive_npms_mileage_features(serv1)
            
            logger.info("Deriving PMS service intervals")
            avg_monthly_interval = derive_pms_service_intervals(serv, last_service_code)
            # avg_monthly_interval.to_csv(f'validatecode/avgmonthlyintervalpmsM{last_service_code}kpreldb.csv', index=False)
            logger.info("Adjusting service intervals")
            avg_monthly_interval1 = adjust_service_intervals(avg_monthly_interval,last_service_code)
            
            logger.info("Calculating NPMS metrics")
            freq_npms, npms_counts, npmsrevenue, pms_counts = derive_npms_features2(serv1)
            Npmsevents,npms_counts = get_non_pms_events(servM, servcode_desc, mastertrain, last_service_code)

            logger.info("Calculating NPMS/PMS Revenue metrics")
            revenu = compute_service_features(servM,filterdate ,last_service_num=last_service_code)

        except Exception as e:
            logger.error(f"Error in feature derivation: {str(e)}")
            raise

        # Log descriptive statistics
        logger.info("Logging descriptive statistics")
        logger.info(f'NPMS counts:\n{npms_counts["nNPMS"].describe()}')
        logger.info(f'NPMS revenue:\n{npmsrevenue["npmsRevenue"].describe()}')
        logger.info(f'PMS counts:\n{pms_counts["nPMS"].describe()}')
        logger.info(f'Freq NPMS:\n{freq_npms["freq_NPMS"].describe()}')

        logger.info("Aggregating service data")
        newserv20k = serv.groupby(['Vin_No']).agg(
            Service_Date=('Service_Date','max'),
            Mileage=('Mileage', 'max'),
            Total_Revenue=('Revenue', 'sum'),
        ).reset_index()
        logger.info(f'UNIQUE VINS in service data: {newserv20k["Vin_No"].nunique()}')

        # Log descriptive statistics
        logger.info(f'NPMS counts:\n{npms_counts["nNPMS"].describe()}')
        logger.info(f'NPMS revenue:\n{npmsrevenue["npmsRevenue"].describe()}')
        logger.info(f'PMS counts:\n{pms_counts["nPMS"].describe()}')
        logger.info(f'Freq NPMS:\n{freq_npms["freq_NPMS"].describe()}')

        # Branch features
        logger.info("Calculating branch-related features")
        vin_branch_pivot,top_branches = branch_visit_features(serv, last_service_code)
        branch_grouped = branch_diversity_features(serv, last_service_code)

        # appoinshow = pd.read_csv(appointshow, low_memory=False) #Input appointment show data
        # appoindf = map_appointments_to_services(appoinshow, servM) 
        # finalappoint = build_final_pms_appointment_summary(appoindf, mastertrain[['VIN','Service_Num']])
        # fnlappnt = adjust_for_target_pms(finalappoint, appoindf, last_service_code)
        # appoinoshow = pd.read_csv(appointnoshow, low_memory=False) #Input appointment no-show data
        # fnlappoinoshow = compute_no_show_appointments(
        #     master_df= mastertrain[['VIN','Last Service Date - PMS','TargetFlag']],
        #     noshow_df=appoinoshow,
        #     filter_date=filterdate,   
        #     vin_col="VIN",
        #     targetflag_col="TargetFlag",
        #     last_pms_date_col="Last Service Date - PMS",
        #     wip_deleted_col="WIP Deleted"
        # )
        complaint_features = transform_complaint_features(mastertrain[['VIN','Service_Num']], servM)
        # fnlappnt.to_csv('validatecode/finalappointpms40kpred.csv', index=False)
        # fnlappoinoshow.to_csv('validatecode/finalnoshowpms40kpred.csv', index=False)
        # complaint_features.to_csv('validatecode/complaintfeatures40kpred.csv', index=False)
        avg_monthly_interval1 = avg_monthly_interval1.rename(columns = {
            "Avg_Service_Interval_PMS": "Avg_Service_Interval_PMSold",
            "Avg_Service_Interval_PMS1": "Avg_Service_Interval_PMS1old",
            "Avg_Service_Interval_PMSnew": "Avg_Service_Interval_PMSper10k",
            "Avg_Service_Interval_PMS1new": "Avg_Service_Interval_PMS"
                })
        # avg_monthly_interval1.to_csv(f'validatecode/avgmonthlyintervalpms{last_service_code}kpred.csv', index=False)
        # Merging all features
        logger.info("Merging all features")
        try:
            newserv20ka = newserv20k.copy()
            # Unpack the tuple returned from derive_npms_features

            merge_operations = [
                (npms_counts, 'NPMS counts'),
                (Npmsevents, 'NPMS events'),
                (dfpmsdate, 'PMS dates'),
                (dfnpmsdate, 'NPMS dates'),
                (avg_mileage_interval, 'Mileage intervals PMS'),
                (avg_mileage_interval_non_pms, 'Mileage intervals NPMS'),
                (npmsrevenue, 'NPMS revenue'),
                (revenu, 'PMS/NPMS revenue'),
                (vin_branch_pivot, 'Branch visits'),
                (avg_monthly_interval1, 'Monthly intervals'),
                (freq_npms, 'NPMS frequency'),
                (branch_grouped, 'Branch diversity'),
                (lastnonpmsmil, 'Last Non-PMS mileage'),
                # (fnlappnt, 'Service Appointment'),
                # (fnlappoinoshow, 'No-show Appointments'),
                (complaint_features, 'Complaint features')
            ]

            
            for df, description in merge_operations:
                print(f"Merging {description}: {df['Vin_No'].is_unique}")

                logger.info(f"Merging {description}")
                before_shape = newserv20ka.shape
                print(f'Merging {description}, Type: {type(df)}')
                newserv20ka = pd.merge(newserv20ka, df, on=['Vin_No'], how='left')
                after_shape = newserv20ka.shape
                logger.debug(f"Shape after merging {description}: {after_shape}")
                if before_shape[0] != after_shape[0]:
                    logger.warning(f"Row count changed after merging {description}")

            # Fill missing values
            logger.info("Handling missing values")
            fill_columns = ['nNPMS', 'npmsRevenue', 'freq_NPMS', 'unique_branch_serviced','otherbranch_services','SinglePMS',
                            'has_complaint_history','num_past_complaints','avg_complaint_resolution_days','max_complaint_resolution_days',
                            'recent_complaint_resolution_days'] + top_branches
            newserv20ka[fill_columns] = newserv20ka[fill_columns].fillna(0)
            newserv20ka[revenu.columns[1:]] = newserv20ka[revenu.columns[1:]].fillna(0)
            # Final processing
            logger.info("Performing final data transformations")
            newserv20ka = newserv20ka.drop(['Total_Revenue','PMS_Service'], axis=1,errors='ignore')
            newserv20ka = newserv20ka.rename(columns={'Vin_No': 'VIN'})
            
            # Merge with master train data
            logger.info("Merging with master train data")
            filtered_dfnew = pd.merge(mastertrain, newserv20ka, on=['VIN'], how='left')
            filtered_dfnew[fill_columns] = filtered_dfnew[fill_columns].fillna(0)
            filtered_dfnew['latest_complaint_category'] = filtered_dfnew['latest_complaint_category'].fillna('unknown') 
            # Save output
            # output_path = 'validatecode/finalmerged40k.csv'
            # logger.info(f"Saving final output to {output_path}")
            filtered_dfnew = filtered_dfnew[filtered_dfnew['VIN'].isin(mastertrain['VIN'].unique())]
            filtered_dfnew[filtered_dfnew["Service_Num_x"] == filtered_dfnew["Service_Num_y"]]
            filtered_dfnew = filtered_dfnew.drop_duplicates(subset=['Vehicle Key'])
            filtered_dfnew = filtered_dfnew.drop(['Service_Num_y','Branch_List'], axis=1)
            filtered_dfnew = filtered_dfnew.rename(columns={'Service_Num_x': 'Service_Num'})
            
            pmsvins = mastertrain.query("TargetFlag == 1")['VIN'].unique()
            service_histories = serv[serv['Vin_No'].isin(pmsvins)].to_dict(orient="records")
            result = find_previous_service_for_all_vins(service_histories, target_service_num=last_service_code)
            cleaned = {vin: rec for vin, rec in result.items() if rec is not None}
            unclean = {vin: rec for vin, rec in result.items() if rec is None}
            finlmlg = pd.DataFrame.from_dict(cleaned, orient='index').reset_index(drop=True)
            finlmlg = finlmlg[['Vin_No','Mileage']]
            pmsfsttime = (pd.DataFrame.from_dict(unclean, orient="index", columns=["Mileage"])
            .reset_index()
            .rename(columns={"index": "Vin_No"}))
            pmsfsttime['Mileage'] = pmsfsttime['Mileage'].fillna(finlmlg['Mileage'].mean())
            lstmiltg = pd.concat([finlmlg,pmsfsttime],axis=0)
            lstmiltg = lstmiltg.rename(columns={'Mileage':'MileagePMS','Vin_No':'VIN'})
            # lstmiltg.to_csv('validatecode/lastpmstimeservicemileage40k.csv', index=False)
            filtered_dfnew.loc[filtered_dfnew["TargetFlag"] == 1, "Last PMS Mileage"] = pd.NA
            filtered_dfnew = filtered_dfnew.merge(lstmiltg,on='VIN',how='left')
            filtered_dfnew["Last PMS Mileage"] = filtered_dfnew["Last PMS Mileage"] \
                                      .fillna(filtered_dfnew["MileagePMS"])
            filtered_dfnew["LastNonPMSMileage"] = filtered_dfnew["LastNonPMSMileage"] \
                                     .fillna(filtered_dfnew["Last PMS Mileage"])
            # filtered_dfnew.to_csv(f'validatecode/checkdf{last_service_code}kpred.csv', index=False)
            # serv.to_csv(f'validatecode/servfiltered{last_service_code}kpred.csv', index=False)
            # servM.to_csv(f'validatecode/servMfiltered{last_service_code}kpred.csv', index=False)
            # filtered_dfnew = map_vhc_history(filtered_dfnew, serv) 
            # vhcfill = ['Survey Score','VHC Quoted','VHC Sold','VHC Lost Sale','VHC Lost Red Sale','VHC Deferred','VHC Amber Deferred','VHC Completed_Flag']
            # filtered_dfnew[vhcfill] = filtered_dfnew[vhcfill].fillna(0)
            # filtered_dfnew = backfill_vhc_for_target_rows(
            #         merged_df=filtered_dfnew,
            #         history_df=servM.query("Service_Num > 0"),
            #         target_service_num=last_service_code,
            #         vhc_cols=["VHC Quoted", "VHC Sold", "VHC Lost Sale","VHC Lost Red Sale", "VHC Deferred", "VHC Amber Deferred"],
            #         survey_cols=["Survey Score", "Survey Status", "VHC Completed_Flag"]
            #     )
            # filtered_dfnew = filtered_dfnew.drop(vhcfill, axis=1, errors='ignore')
            vhcdf = pd.read_csv(vhc,encoding="ISO-8859-1",low_memory=False)
            vhcdata = vhcpreparation(vhcdf,last_service_code,filtered_dfnew)
            filtered_dfnew = filtered_dfnew.merge(vhcdata,on=['Vehicle Key','Service_Num'],how='left')
            filtered_dfnew = backfill_vhc_leakage_safe(
            filtered_dfnew,
            vhcdata,
            vhcdata.columns[2:],
            vin_col="Vehicle Key",
            service_col="Service_Num",
            targetflag_col="TargetFlag"
            )
            filtered_dfnew = filtered_dfnew.fillna(0)
            # filtered_dfnew.to_csv(f'validatecode/checkdfwithvhc{last_service_code}kpred.csv', index=False)
            
            appointdfN = pd.read_csv(appointdf, low_memory=False) #Input appointment full data
            appointdfN['Due Date IN'] = pd.to_datetime(appointdfN['Due Date IN'],format='mixed',dayfirst=True,errors='coerce')

            ltappoins = compute_late_appointment_metrics(appointdfN, servM,filter_date=filterdate)
            appoinstat = compute_last_appointment_status_with_constant_service_code(appointdfN,serv,last_service_code,filter_date=filterdate)
            fnlappnt = derive_appointment_show_features(appointdfN,servM,last_service_code,filter_date=filterdate)
            filtered_dfnew["Vehicle Magic"] = filtered_dfnew["Vehicle Key"].str.split("-", n=1).str[1]
            filtered_dfnew["Vehicle Magic"] = filtered_dfnew["Vehicle Magic"].str.split("-", n=1).str[0]
            filtered_dfnew["Vehicle Magic"] = filtered_dfnew["Vehicle Magic"].astype(int)
            
            filtered_dfnew = filtered_dfnew.merge(appoinstat, on='Vehicle Magic',how='left').merge(ltappoins,on='Vehicle Magic',how='left').merge(fnlappnt,on='Vehicle Magic',how='left')
            nservappbk = count_service_appointments_booked(
                        appointdfN,
                        filtered_dfnew,
                        last_service_code,
                        filterdate,
                        vehicle_col="Vehicle Magic",
                        booking_date_col="WIP Booking Date"
                    )
            filtered_dfnew = filtered_dfnew.merge(nservappbk,on='Vehicle Magic',how='left')
            filtered_dfnew['no_of_service_appointments_booked'] = filtered_dfnew['no_of_service_appointments_booked'].fillna(0)
            # filtered_dfnew.query("no_of_service_appointments_booked == 0 and total_appointments_showed_up > 0").to_csv('validatecode/checknoofserviceappbooked.csv', index=False)
            # ltappoins.to_csv('validatecode/lateappointments60kpred.csv', index=False)
            # appoinstat.to_csv('validatecode/lastappointmentstatus60kpred.csv', index=False)
            filtered_dfnew = filtered_dfnew.drop(['Vehicle Magic'],axis=1)
            # filtered_dfnew["Vehicle Magic"] = filtered_dfnew["Vehicle Key"].str.split("-", n=1).str[1]
            # filtered_dfnew["Vehicle Magic"] = filtered_dfnew["Vehicle Magic"].str.split("-", n=1).str[0]
            # filtered_dfnew["Vehicle Magic"] = filtered_dfnew["Vehicle Magic"].astype(int)

            filtered_dfnew[['last_appointment_delay_days', 'no_of_late_appointments','appointment_booked_showed_up_atleastonce','total_appointments_showed_up','no_of_appointments_booked_but_not_showed_up','Avg_appointment_delay_days']] = filtered_dfnew[['last_appointment_delay_days', 'no_of_late_appointments','appointment_booked_showed_up_atleastonce','total_appointments_showed_up','no_of_appointments_booked_but_not_showed_up','Avg_appointment_delay_days']].fillna(0)
            filtered_dfnew['last_appointment_status'] = filtered_dfnew['last_appointment_status'].fillna('No Appointment')
            #last_appointment_status	last_appointment_delay_days
            # filtered_dfnew.loc[
            #         filtered_dfnew["Service_Num"] == 10,
            #         "Avg_Mileage_Interval_PMS"
            #     ] = (((last_service_code * 1000) - filtered_dfnew.loc[
            #         filtered_dfnew["Service_Num"] == 10, "Last PMS Mileage"
            #     ]) + (filtered_dfnew.loc[
            #         filtered_dfnew["Service_Num"] == 10, "Last PMS Mileage"
            #     ])) /2
            # filtered_dfnew.loc[filtered_dfnew['Service_Num'] == 10, 'Avg_Mileage_Interval_PMS'] = filtered_dfnew.loc[filtered_dfnew['Service_Num'] == 10, 'Avg_Mileage_Interval_PMSold']
            logger.info(f"Processing completed. Final shape: {filtered_dfnew.shape}")
      
            # filtered_dfnew = pd.read_csv('validatecode/finalmerged40k.csv',low_memory=False)
            rfmdf = pd.read_csv(rfm,low_memory=False,encoding='ISO-8859-1') #Input RFM segments file
            filtered_dfnew = filtered_dfnew.merge(rfmdf[['Customer ID','RFM_segments']],on=['Customer ID'],how='left')
            filtered_dfnew['Number of Cylinders'] = filtered_dfnew['Number of Cylinders'].astype('object')
            filtered_dfnew = filtered_dfnew.rename(columns={f'Expected{last_service_code}kDate':f'Next{last_service_code}K_Due'})
            rem = ['New / Used Category','Current Customer','First Service Date',
                'Last Service Date', 'Next Service Date','Vehicle Lifetime in Years','MileagePMS',
                'Last Service - PMS','Sale Invoice Year','Invoice date','Target Revenue','Potential Revenue',
                  'Final Revenue', 'Vehicle Age']
            filtered_dfnew= filtered_dfnew.drop(rem,axis=1,errors='ignore')
            
            # Compute PMS_Delay accurately
            filtered_dfnew['PMS_Delay'] = compute_pms_delay(filtered_dfnew, last_service_code)
            
            pms = filtered_dfnew
            logger.info("Detecting columns with '-' placeholders (pass 1)")
            obj_cols = pms.select_dtypes(include='object').columns
            hifunfeats = [c for c in obj_cols if (pms[c] == '-').any()]
            missfeats = [c for c in pms.columns if c not in hifunfeats and pms[c].isnull().any()]
            logger.info(f"Pass 1: {len(hifunfeats)} cols with '-', {len(missfeats)} cols with nulls")
            cond1 = pms['Brake Purchase Interval'] > 0
            cond2 = pms['Tyre Purchase Interval'] > 0
            cond3 = pms['Battery Purchase Interval'] > 0
            pms['wearablesBought'] = np.where(cond1 | cond2 | cond3,1,0)
            pms = calculate_revenue_spend(pms,last_service_code)
            # filtered_dfnew.to_csv(output_path, index=False)
            missfeatsdrop = ['10K','20K','30K','40K','50K','60K','70K','80K','90K','100K','110K',
                '10K R','20K R','30K R','40K R','50K R','60K R','70K R','80K R','90K R','100K R','110K R',
                '10K SC','20K SC','30K SC','40K SC','50K SC','60K SC','70K SC','80K SC','90K SC','100K SC',
                '110K SC', '120K', '120K R', '120K SC', '130K', '130K R', '130K SC', '140K', '140K R','140K SC',
                '150K', '150K R', '150K SC', '160K', '160K R', '160K SC', '170K', '170K R', '170K SC',
                '180K', '180K R', '180K SC', '190K', '190K R', '190K SC', '200K', '200K R', '200K SC',
                '200K+', '200K+   R', '200K+   SC','Brake Purchase Interval','Tyre Purchase Interval',
                'Battery Purchase Interval','70K R.1']
            pms = pms.drop(missfeatsdrop,axis=1,errors = 'ignore')

            logger.info("Detecting columns with '-' placeholders (pass 2)")
            obj_cols2 = pms.select_dtypes(include='object').columns
            hifunfeats1 = [c for c in obj_cols2 if (pms[c] == '-').any()]
            missfeats1 = [c for c in pms.columns if c not in hifunfeats1 and pms[c].isnull().any()]
            logger.info(f"Pass 2: {len(hifunfeats1)} cols with '-', {len(missfeats1)} cols with nulls")
            
            for i in hifunfeats1:
                pms[i] = pms[i].replace('-', pd.NA)

            # Convert '-'-cleaned columns to numeric ONLY if their non-null
            # values are actually numeric (preserves categoricals like
            # New/Used, Gender, Warranty Status that also had '-' placeholders)
            converted_count = 0
            for col in hifunfeats1:
                test = pd.to_numeric(pms[col], errors='coerce')
                non_null_orig = pms[col].notna().sum()
                if non_null_orig > 0 and test.notna().sum() / non_null_orig > 0.5:
                    pms[col] = test
                    converted_count += 1
                else:
                    logger.info(f"  Kept '{col}' as categorical (not numeric)")
            logger.info(f"Converted {converted_count}/{len(hifunfeats1)} columns to numeric after '-' cleanup")

            print(f'Dtype of Vehicle_Key_ExpectedServices: {pms["Vehicle_Key_ExpectedServices"].dtype}')
            catfeats = ['New / Used','Gender','Warranty Status', 'Number of Cylinders']
            for i in ['Nationality','Model','Variant','RFM_segments']:
                pms[i] = pms[i].fillna('Unknown')
            nationalitymap = pms[['Vehicle Key','Nationality']]
            vehmodelmap = pms[['Vehicle Key','Model']]
            vehvariantmap = pms[['Vehicle Key','Variant']]
            rfmpms= pms[['Vehicle Key','RFM_segments']]
            # nationalitymap.to_csv('validatecode/nationalitymap.csv', index=False)
            # vehmodelmap.to_csv('validatecode/vehmodelmap.csv', index=False)
            # vehvariantmap.to_csv('validatecode/vehvariantmap.csv', index=False)
            # rfmpms.to_csv('validatecode/rfmpms.csv', index=False)
            obint = ['Total Promoter','Total Passive','Total Detractor','Total Survey',
            'CC','Weight','Height','Wheel Base','Service Frequency']
            for i in obint:
                pms[i] = pd.to_numeric(pms[i], errors='coerce')
            pms = pms.drop('Service_Date',axis=1,errors = 'ignore')
            pms[['Months_Since_Last_NPMS','Avg_Mileage_Interval_NPMS']] = pms[['Months_Since_Last_NPMS','Avg_Mileage_Interval_NPMS']].fillna(0)
            colssrvd = pms.pop('Last Service Date - PMS')
            pms.insert(3, colssrvd.name, colssrvd)
            pms['Last Service Date - PMS'] = pd.to_datetime(pms['Last Service Date - PMS'],format='mixed',dayfirst=True,errors='coerce')
            pms[f'Next{last_service_code}K_Due'] = pd.to_datetime(pms[f'Next{last_service_code}K_Due'],format='mixed',dayfirst=True,errors='coerce')
            colsdue = pms.pop(f'Next{last_service_code}K_Due')
            pms.insert(4, colsdue.name, colsdue)
            # Drop high-cardinality columns before encoding (already saved for clustering)
            pms = pms.drop(['Nationality','Model','Variant','RFM_segments'],axis=1,errors='ignore')
            # pms.dtypes.reset_index().rename(
            #     columns={"index": "feature", 0: "dtype"}
            # ).to_csv('validatecode/pms_dtypes.csv', index=False)
            # pms.to_csv('validatecode/pms_before_onehot.csv', index=False)
            # pms.to_csv(f'validatecode/pms_before_onehot{last_service_code}k.csv', index=False)
            # numeric_cols = [
            #     "Survey Score",
            #     # "VHC Quoted",
            #     # "VHC Sold",
            #     # "VHC Lost Sale",
            #     # "VHC Lost Red Sale",
            #     # "VHC Deferred",
            #     # "VHC Amber Deferred"
            # ]

            # pms[numeric_cols] = pms[numeric_cols].apply(
            #     lambda col: pd.to_numeric(col, errors="coerce")
            # )
            # ── Robust one-hot encoding ─────────────────────────────
            ID_COLS = {'VIN', 'Vehicle Key', 'Customer ID'}
            MAX_OHE_CARDINALITY = 50  # safety cap

            # 1) Try numeric conversion on every non-ID object column
            for col in pms.select_dtypes(include='object').columns:
                if col in ID_COLS:
                    continue
                converted = pd.to_numeric(pms[col], errors='coerce')
                if converted.notna().mean() > 0.5:  # >50% parseable → numeric
                    pms[col] = converted
                    logger.info(f"  Coerced '{col}' to numeric ({converted.notna().mean():.0%} parseable)")

            # 2) Identify true categoricals (non-ID, low-cardinality)
            obj_cols_remaining = [
                c for c in pms.select_dtypes(include='object').columns
                if c not in ID_COLS
            ]
            cols_to_encode = [c for c in obj_cols_remaining if pms[c].nunique() <= MAX_OHE_CARDINALITY]
            cols_too_high  = [c for c in obj_cols_remaining if pms[c].nunique() > MAX_OHE_CARDINALITY]

            for c in cols_to_encode:
                logger.info(f"  OHE: {c} ({pms[c].nunique()} unique values)")
            if cols_too_high:
                logger.warning(f"Dropping high-cardinality object columns (>{MAX_OHE_CARDINALITY} unique): {cols_too_high}")
                pms = pms.drop(columns=cols_too_high)

            # 3) Encode
            logger.info(f"One-hot encoding {len(cols_to_encode)} columns")
            pmsnew = pd.get_dummies(pms, columns=cols_to_encode, dtype=int)
            pmsnew = pmsnew.reset_index(drop=True)
            logger.info(f'One-hot encoding complete. Shape: {pmsnew.shape}')
            
            pmsnew["LowMileageFreqUsers"] = np.where(
             (pmsnew["Avg_Service_Interval_PMS"].between(0, 7)) & (pmsnew["Last Service Mileage"].between(0, (last_service_code-10)*1000)) & (pmsnew["TargetFlag"]==1),1,0)
            uy = ['Total Promoter','Total Passive','Total Detractor','Total Survey']
            for i in uy:
                pmsnew[i] = pmsnew[i].fillna(0)
            cols = pmsnew.pop('Last Service Date - PMS')
            pmsnew.insert(1, cols.name, cols)
            pmsnew = pmsnew.drop('70K R.1',axis=1,errors='ignore')
            # pmsnew.to_csv(f'validatecode/finalmerged{last_service_code}kv1.csv', index=False)
            # Fill all-NaN columns with 0 before imputing (IterativeImputer
            # silently drops them, causing a shape mismatch on reconstruction)
            numeric_block = pmsnew.iloc[:, 5:]
            all_nan_cols = numeric_block.columns[numeric_block.isnull().all()]
            if len(all_nan_cols) > 0:
                logger.warning(f"Filling {len(all_nan_cols)} all-NaN columns with 0 before imputation: {list(all_nan_cols)}")
                pmsnew[all_nan_cols] = 0
            imputer = IterativeImputer(random_state=0,estimator=Lasso(), max_iter=1)
            pms_imputed = imputer.fit_transform(pmsnew.iloc[:,5:])
            pms_imputed = pd.DataFrame(pms_imputed, columns=pmsnew.columns[5:])
            pms_imputed['VIN'] = pmsnew['VIN'].values
            col1 = pms_imputed.pop("VIN")
            pms_imputed.insert(0, col1.name, col1)
            pms_imputed['Last Service Date - PMS'] = pmsnew['Last Service Date - PMS'].values
            cols2 = pms_imputed.pop('Last Service Date - PMS')
            pms_imputed.insert(1, cols2.name, cols2)
            pms_imputed['Vehicle Key'] = pmsnew['Vehicle Key'].values
            col = pms_imputed.pop("Vehicle Key")
            pms_imputed.insert(2, col.name, col)
            pms_imputed['Customer ID'] = pmsnew['Customer ID'].values
            cols3 = pms_imputed.pop("Customer ID")
            pms_imputed.insert(3, cols3.name, cols3)
            pms_imputed[f'Next{last_service_code}K_Due'] = pmsnew[f'Next{last_service_code}K_Due'].values
            cols4 = pms_imputed.pop(f"Next{last_service_code}K_Due")
            pms_imputed.insert(4, cols4.name, cols4)
            bs = serv1[serv1['Description']=='Others']
            bodyshop_counts = calculate_bodyshop_count(bs, branch_col='Service_Branch_Name', vin_col='Vin_No', keyword='Bodyshop')
            pms_imputed = pms_imputed.merge(bodyshop_counts,on='VIN',how='left')
            pms_imputed['Bodyshop_Services'] = pms_imputed['Bodyshop_Services'].fillna(0).astype(int)
            # pms_imputed.to_csv('validatecode/finalmerged40kv2.csv', index=False)
            ##Nationality
            mergedfnat = nationalitymap.merge(vehmodelmap,on='Vehicle Key',how='left')
            mergedfnat = mergedfnat.merge(rfmpms,on='Vehicle Key',how='left')
            pms_imputed, clustersnat = hierarchical_gower_clustering(pms_imputed,mergedfnat,
                                                            feat="Nationality", k_min=2, k_max=10)
            
            ##Model
            mergedfmod = vehmodelmap.merge(rfmpms,on='Vehicle Key',how='left')
            pms_imputed, clustersmod = hierarchical_gower_clustering(pms_imputed,mergedfmod,
                                                feat="Model", k_min=2, k_max=10)
            
            ##Variant
            mergedfvar = vehvariantmap.merge(rfmpms,on='Vehicle Key',how='left')
            pms_imputed, clustersvar = hierarchical_gower_clustering(pms_imputed,mergedfvar,
                                                feat="Variant", k_min=2, k_max=10)
            pms_imputed = pms_imputed.drop(['Mileage','Brake Points','Vehicle_Key_ExpectedServices',
                                            'Tyre Points','Battery Points'],axis=1,errors='ignore')
            
            clustersnat.to_csv(f'validatecode/Nationality_clusters_{last_service_code}.csv', index=False)
            clustersmod.to_csv(f'validatecode/Model_clusters_{last_service_code}.csv', index=False)
            clustersvar.to_csv(f'validatecode/Variant_clusters_{last_service_code}.csv', index=False)
            sessdf = pd.read_csv(digidf,low_memory=False)
            pms_imputed = pms_imputed.merge(sessdf,on=['Vehicle Key','Customer ID'],how='left')
            pms_imputed.iloc[:,-4:] = pms_imputed.iloc[:,-4:].fillna(0)
            pms_imputed['Vehicles Owned'] = pms_imputed['Vehicles Owned'].fillna(1)
            coltarg = pms_imputed.pop("TargetFlag")      
            pms_imputed.insert(len(pms_imputed.columns), "TargetFlag", coltarg) 
            coltpmsmil = pms_imputed.pop("Last PMS Mileage")      
            pms_imputed.insert(len(pms_imputed.columns)-1, "Last PMS Mileage", coltpmsmil)
            # pms_imputed.rename(columns={'Avg_Service_Interval_PMS':'Avg_Service_Interval_PMS_per10k'},inplace=True)
            # pms_imputed.rename(columns={'Avg_Service_Interval_PMS1':'Avg_Service_Interval_PMS'},inplace=True)
            pms_imputed = pms_imputed.rename(columns=lambda c: re.sub(r'\.0$', '', c))
            pms_imputed["Last Service Mileage"] = pms_imputed[["Last PMS Mileage", "LastNonPMSMileage"]].max(axis=1)
            os.makedirs(f'models/{last_service_code}k', exist_ok=True)
            pms_imputed.to_csv(f'models/{last_service_code}k/training_features.csv', index=False)
            pms_imputed.to_csv(f'validatecode/finalmerged{last_service_code}kQ3.csv', index=False)  # legacy compat
            # pms_imputed.to_csv('validatetrainnomseventrev.csv', index=False)
            return pms_imputed
        

if __name__ == "__main__":
    import sys
    import os

    dataframe_result = process_service_data()
