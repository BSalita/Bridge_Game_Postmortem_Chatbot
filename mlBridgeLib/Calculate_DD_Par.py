
# Assume all ACBL double dummy and par calculations are wrong! Recompute both.

import mlBridgeLib

import dds
import ctypes
import functions


# valdiate calculated par result by comparing against known (assumed) correct par. Similar to functions.ComparePar().
def ComparePars(par1, par2):
    if par1[0] != par2[0]:
        return False
    if len(par1[1]) != len(par2[1]):
        return False
    for p1,p2 in zip(sorted(par1[1],key=lambda k: k[1]),sorted(par2[1],key=lambda k: k[1])):
        if p1 != p2:
            if p1[1] != p2[1] or p1[2] != p2[2] or p1[3] != p2[3]:
                return False # suit/double/direction difference
            if p1[0]+p1[4] != p2[0]+p2[4]:
                return False # only needs to have trick count same. ignore if levels are same (min level+overs vs max level+0)
    return True


# valdiate calculated dd result by comparing against known (assumed) correct dd. Similar to functions.CompareTable().
def CompareDDTables(DDtable1, DDtable2):
    #print(DDtable1,DDtable2)
    for suit in range(dds.DDS_STRAINS):
        for pl in range(4):
            if DDtable1[pl][suit] != DDtable2[pl][suit]:
                return False
    return True


def Calculate_DD_Par(df, d):

    DDdealsPBN = dds.ddTableDealsPBN()
    tableRes = dds.ddTablesRes()
    pres = dds.allParResults()
    presmaster = dds.parResultsMaster()

    mode = 0
    tFilter = ctypes.c_int * dds.DDS_STRAINS
    trumpFilter = tFilter(0, 0, 0, 0, 0)
    line = ctypes.create_string_buffer(80)

    dds.SetMaxThreads(0)
    max_tables = dds.MAXNOOFTABLES
    nhands = len(df)

    mismatched_dds = []
    mismatched_pars = []
    
    for n,grp_start in enumerate(range(0,nhands,max_tables)):
        n += 1
        grp_count = min(nhands-grp_start,max_tables)
        DDdealsPBN.noOfTables = grp_count
        print(f"Processing group:{n} hands:{grp_start} to {grp_start+grp_count} of {nhands} dict len:{len(d)}")
        indexes = df.iloc[grp_start:grp_start+grp_count].index
        for handno in range(grp_count):
            r = df.loc[indexes[handno]]
            # create lists of PBN
            nesw_tuple = r['Hands']
            pbn = 'N:'+' '.join('.'.join([suit for suit in suits]) for suits in nesw_tuple)
            assert pbn == mlBridgeLib.HandToPBN(r['Hands']), [pbn,mlBridgeLib.HandToPBN(r['Hands'])] # todo: use HandToPBN() instead
            assert r['Hands'] == mlBridgeLib.pbn_to_hands(pbn)
            assert len(pbn) == 1+1+13*4+3*4+3, len(pbn) # 69=='N'+':'+(13 cards per hand)+(3 '.' per suit per hand)+3 spaces
            pbn_encoded = pbn.encode() # requires pbn to be utf8 encoded.
            #print(len(pbn),pbn,pbn_encoded)
            DDdealsPBN.deals[handno].cards = pbn_encoded

        # CalcAllTablesPBN will do multi-threading
        res = dds.CalcAllTablesPBN(ctypes.pointer(DDdealsPBN), mode, trumpFilter, ctypes.pointer(tableRes), ctypes.pointer(pres))

        if res != dds.RETURN_NO_FAULT:
            dds.ErrorMessage(res, line)
            print(f"CalcAllTablesPBN: DDS error: {line.value.decode('utf-8')}")
            assert False, grp_start

        for handno in range(0, grp_count):
            r = df.loc[indexes[handno]]
            # shortcut to avoid recompute. only compute when either DDmakes or Par is None. Assumes Hands are everywhere correct.
            if r['DDmakes'] is not None and r['Par'] is not None:
                if r['board_record_string'] in d: # assumes all d entries are valid.
                    if r['DDmakes'] != d[r['board_record_string']]['DDmakes']:
                        print([r['hand_record_id'],r['board_record_string'],r['Hands'], r['DDmakes'],d[r['board_record_string']]['DDmakes']])
                        d[r['board_record_string']] = {'DDmakes':r['DDmakes'],'Par':r['Par'],'Hands':r['Hands']}
                        d[r['HandRecordBoard']] = r['board_record_string']
                    #    continue
                    assert r['DDmakes'] == d[r['board_record_string']]['DDmakes'], [r['hand_record_id'],r['board_record_string'],r['Hands'], r['DDmakes'],d[r['board_record_string']]['DDmakes']]
                    #assert r['Par'] == d[r['board_record_string']]['Par'],[r['Par'] == d[r['board_record_string']]['Par']] # Hand ('acbl', 533564) had wrong vul but it's 13-0-0-0 hand so weird anyway.
                    assert r['Hands'] == d[r['board_record_string']]['Hands']
                    if r['Par'] == d[r['board_record_string']]['Par']: # temp fix for wrong vuls
                        continue # dict entry seems correct. no need to compute. skipping.
                else:
                    # assumes df has valid DDmakes and Par. Is this a valid assumption?
                    assert isinstance(r['DDmakes'],tuple)
                    assert isinstance(r['Par'],tuple)
                    assert isinstance(r['Hands'],tuple)
                    d[r['board_record_string']] = {'DDmakes':r['DDmakes'],'Par':r['Par'],'Hands':r['Hands']}
                    d[r['HandRecordBoard']] = r['board_record_string']
                    continue

            # compute double dummy
            dealer = r['Dealer'] # todo: change to 'Dealer_Direction' like Morty ?
            assert dealer in mlBridgeLib.NESW, r
            vul = r['Vul']
            assert vul in ['None','N_S','E_W','Both'], r
            par_result = ctypes.pointer(presmaster)
            dd_result = ctypes.pointer(tableRes.results[handno])

            # Par calculations are not multi-threading
            res = dds.DealerParBin(dd_result, par_result, mlBridgeLib.NESW.index(dealer), mlBridgeLib.vul_d[vul])
            if res != dds.RETURN_NO_FAULT:
                dds.ErrorMessage(res, line)
                print(f"DealerParBin: DDS error: {line.value.decode('utf-8')}")
                assert False, r

            DDtable_solved = tuple(tuple(dd_result.contents.resTable[suit][pl] for suit in [3,2,1,0,4]) for pl in range(4))

            score = par_result.contents.score
            if score == 0:
                par_solved = (0, [(0, '', '', '', 0)]) # par score is for everyone to pass (1 out of 100,000)
            else:
                assert par_result.contents.number > 0, r
                par_solved = (score,[])
                for i in range(par_result.contents.number):
                    ct = par_result.contents.contracts[i]    
                    #print(f"Par[{i}]: underTricks:{ct.underTricks} overTricks:{ct.overTricks} level:{ct.level} denom:{ct.denom} seats:{ct.seats}")
                    assert ct.underTricks == 0 or ct.overTricks == 0
                    par_solved[1].append((ct.level,mlBridgeLib.NSHDC[ct.denom],'*' if ct.underTricks else '',mlBridgeLib.seats[ct.seats],ct.overTricks-ct.underTricks))

            DDtable = r['DDmakes']
            if DDtable is None:
                DDtable = DDtable_solved
            assert isinstance(DDtable,tuple), type(DDtable)
            assert len(DDtable)==4, DDtable
            assert all([isinstance(t,tuple) and len(t)==5 for t in DDtable]), DDtable
            dd_eq = CompareDDTables(DDtable, DDtable_solved)

            par = r['Par']
            if par is None:
                par = par_solved
            assert type(par) is tuple, r
            assert len(par) == 2, r
            assert type(par[0]) is int, r
            assert type(par[1]) is list, r
            assert len(par[1]) > 0, r
            par_eq = ComparePars(par, par_solved)

            if not dd_eq or not par_eq:
                functions.PrintPBNHand(f"Group:{n} Hand:{grp_start+handno+1}/{nhands} ACBL HRB:{r['HandRecordBoard']} Vul:{vul}", DDdealsPBN.deals[handno].cards)
                functions.PrintTable(dd_result)
                if not dd_eq:
                    print(f"DD mismatch: Current:{DDtable} Computed:{DDtable_solved}")
                    mismatched_dds.append(r['board_record_string'])
                if not par_eq:
                    #functions.PrintDealerParBin(par_result)
                    if par[0] != par_solved[0]: # differences in par score are usually due to vul-unknown vs vul-known.
                        print(f"Par score mismatch: Vul ignorance yielding wrong score?: Current:{par[0]} Calculated:{par_solved[0]}")
                    else:
                        print(f"Par contract mismatch: (Vul ignorance effecting interfering bid?): Current:{par[1]} Calculated:{par_solved[1]}")
                    mismatched_pars.append(r['board_record_string'])
                print()

            d[r['board_record_string']] = {'DDmakes':DDtable_solved,'Par':par_solved,'Hands':r['Hands']}
            d[r['HandRecordBoard']] = r['board_record_string']

    print(f"mismatched_dds: len:{len(mismatched_dds)} {mismatched_dds}")
    print(f"mismatched_pars: len:{len(mismatched_pars)} {mismatched_pars}")
    return mismatched_dds, mismatched_pars
