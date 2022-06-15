# FIN6

fin6_tecs_intel = ['T1078', 'T1133', 'T1566', ' T1046', 'T1069', 'T1068', 'T1018', 'T1105', 'T1553', 'T1547', 'T1053', 'T1021', 
            'T1059', 'T1047', 'T1003', 'T1048', 'T1005', 'T1027', 'T1560', 'T1074', 'T1190', 'T1072', 'T1080', 'T1484']

fin6_tec_2 = ['T1134', 'T1059', 'T1562', 'T1036', 'T1588', 'T1003', 'T1021', 'T1569', 'T1078', 'T1102',
        'T1087', 'T1482', 'T1069', 'T1018', 'T1016', 'T1548', 'T1071', 'T1185', 'T1059', 
        'T1543', 'T1132', 'T1005', 'T1001', 'T1140', 'T1573', 'T1068', 'T1083',
        'T1564', 'T1562', 'T1070', 'T1105', 'T1056', 'T1112', 'T1046', 'T1095', 'T1027', 'T1137', 
        'T1003', 'T1069', 'T1057', 'T1055', 'T1572', 'T1572', 'T1090', 'T1012', 'T1620', 'T1021',
        'T1018', 'T1029', 'T1113', 'T1518', 'T1553', 'T1218', 'T1049', 'T1007', 'T1569', 'T1550', 
        'T1078', 'T1047']

fin6_tec_1 = ['T1087', 'T1560', 'T1119', 'T1547', 'T1110', 'T1059', 'T1074', 'T1573', 'T1068', 'T1070', 
        'T1046', 'T1003', 'T1572', 'T1021', 'T1018', 'T1053', 'T1078', 'T1003']

# MenuPass [8]

menuPass_tec_8 = ['T1560', 'T1119', 'T1059', 'T1005', 'T1074', 'T1210', 'T1083', 'T1574', 'T1106', 'T1027', 'T1003', 'T1199', 'T1078', 'T1047']

adFind = ['T1087', 'T1482', 'T1069', 'T1018', 'T1016']

certutil = ['T1140', 'T1105', 'T1553']

quasarRAT = ['T1059', 'T1555', 'T1573', 'T1105', 'T1056', 'T1112', 'T1090', 'T1021', 'T1053', 'T1553', 'T1082', 'T1552', 'T1125']

menuPass_tec_8.extend(adFind)
menuPass_tec_8.extend(certutil)
menuPass_tec_8.extend(quasarRAT)

# MenuPass [2]

menuPass_tec_2 = ['T1583', 'T1560', 'T1568', 'T1070', 'T1056', 'T1036', 'T1105', 'T1566', 'T1021', 'T1199', 'T1204', 'T1078']

poisonIvy = ['T1010', 'T1547', 'T1059', 'T1543', 'T1005', 'T1074', 'T1573', 'T1105', 'T1056', 'T1112', 'T1027', 
'T1055', 'T1014']

menuPass_tec_2.extend(poisonIvy)

# WizardSpider [2]

wizardSpider_tec_2 = ['T1547', 'T1059', 'T1562', 'T1135', 'T1566', 'T1055', 'T1021', 'T1053', 'T1558', 'T1204', 'T1047']

bloodHound = ['T1087', 'T1560', 'T1059', 'T1482', 'T1615', 'T1106', 'T1201', 'T1069', 'T1018', 'T1033']

cobaltStrike = ['T1548', 'T1134', 'T1087', 'T1071', 'T1197', 'T1185', 'T1059', 'T1043', 'T1543', 'T1132', 'T1005', 'T1001', 'T1030', 'T1140', 'T1573', 'T1203', 'T1068', 'T1083', 'T1564', 'T1562', 'T1070', 'T1105', 'T1056', 'T1112', 'T1026', 'T1106', 'T1046', 'T1135', 'T1095', 'T1027', 'T1137', 'T1003', 'T1069', 'T1057', 'T1055', 'T1572', 'T1090', 'T1012', 'T1620', 'T1021', 'T1018', 'T1029', 'T1113', 'T1518', 'T1553', 'T1218', 'T1016', 'T1049', 'T1007', 'T1569', 'T1550', 'T1078', 'T1047']

empire =  ['T1548', 'T1134', 'T1087', 'T1557', 'T1071', 'T1560', 'T1547', 'T1217', 'T1115', 'T1059', 'T1043', 'T1136', 'T1543', 'T1555', 'T1484', 'T1482', 'T1114', 'T1573', 'T1546', 'T1068', 'T1083', 'T1574', 'T1210', 'T1615', 'T1567', 'T1070',  'T1056', 'T1105', 'T1056', 'T1106', 'T1046', 'T1135', 'T1040', 'T1027', 'T1003', 'T1057', 'T1055', 'T1021', 'T1053', 'T1113', 'T1518', 'T1558', 'T1082', 'T1016', 'T1049', 'T1569', 'T1127', 'T1552', 'T1550', 'T1125', 'T1102', 'T1047']

mimikatz = ['T1134', 'T1098', 'T1547', 'T1555', 'T1003', 'T1207', 'T1558', 'T1552', 'T1550']

ping = ['T1018']

ryuk = ['T1134', 'T1547', 'T1059', 'T1486', 'T1083', 'T1222', 'T1562', 'T1490', 'T0828', 'T1036', 'T1106', 'T1027', 'T1057', 'T1055', 'T1021', 'T1053', 'T1489', 'T1082', 'T1614', 'T1016', 'T1205', 'T1078']

trickBot = ['T1087', 'T1087', 'T1071', 'T1547', 'T1185', 'T1110', 'T1059', 'T1059', 'T1043', 'T1543', 'T1555', 'T1555', 'T1132', 'T1005', 'T1140', 'T1482', 'T1573', 'T1041', 'T1210', 'T1008', 'T1083', 'T1495', 'T1562', 'T1105', 'T1056', 'T1559', 'T1036', 'T1112', 'T1106', 'T1135', 'T1571', 'T1027', 'T1027', 'T1069', 'T1566', 'T1566', 'T1542', 'T1057', 'T1055', 'T1055', 'T1090', 'T1219', 'T1021', 'T1018', 'T1053', 'T1553', 'T1082', 'T1016', 'T1033', 'T1007', 'T1552', 'T1552', 'T1204', 'T1497']

wizardSpider_tec_2.extend(bloodHound)
wizardSpider_tec_2.extend(cobaltStrike)
wizardSpider_tec_2.extend(empire)
wizardSpider_tec_2.extend(mimikatz)
wizardSpider_tec_2.extend(ping)
wizardSpider_tec_2.extend(ryuk)
wizardSpider_tec_2.extend(trickBot)

#WizardSpider [7]

wizardSpider_tec_7 = ['T1087', 'T1059', 'T1048', 'T1210', 'T1562', 'T1027', 'T1021', 'T1018', 'T1489', 'T1518', 'T1558', 'T1082', 'T1569']

adFind = ['T1087', 'T1482', 't1069', 'T1018', 'T1016']

#CobaltStrike

net = ['T1087', 'T1087', 'T1136', 'T1136', 'T1070', 'T1135', 'T1201', 'T1069', 'T1069', 'T1021', 'T1018', 'T1049', 'T1007', 'T1569', 'T1124']

nltest = ['T1482', 'T1018', 'T1016']

#Ping

#Ryuk

wizardSpider_tec_7.extend(adFind)
wizardSpider_tec_7.extend(cobaltStrike)
wizardSpider_tec_7.extend(net)
wizardSpider_tec_7.extend(nltest)
wizardSpider_tec_7.extend(ping)
wizardSpider_tec_7.extend(ryuk)


# ------------------------------------

fin6_files = ['apt_documents/FIN6/Follow The Money-Dissecting the Operations of the Cyber Crime Group FIN6[1].txt', 
                'apt_documents/FIN6/Pick-Six-Intercepting a FIN6 Intrusion, an Actor Recently Tied to Ryuk and LockerGoga Ransomware[2].txt',
                'apt_documents/FIN6/intelligence_summary.txt']

menuPass_files = ['apt_documents/MenuPass/2018_12_20_united_states_v_zhu_hua_indictment[2].txt', 
                'apt_documents/MenuPass/Japan-Linked Organizations Targeted in Long-Running and Sophisticated Attack Campaign[8].txt']

wizardSpider_files = ['apt_documents/WizardSpider/Ryuk’s Return[7].txt',
                     'apt_documents/WizardSpider/Ransomware Activity Targeting the Healthcare and Public Health Sector. Retrieved October 28, 2020[2].txt']

# ------------------------------------