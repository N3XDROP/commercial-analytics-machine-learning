"""Analítica de datos y aprendizaje automático para la actividad comercial."""

# 01. LIBRERÍAS Y CONFIGURACIÓN

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REQUIRED = ['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'Price', 'Customer ID', 'Country']
SEGMENTS = ['Alto valor', 'Leales', 'Recientes de baja frecuencia', 'En riesgo', 'Otros']
CUTS = ['2010-06-01', '2010-09-01', '2010-12-01', '2011-03-01', '2011-06-01', '2011-09-01']
COLUMNS_AUDIT = ['etapa', 'filas', 'diferencia_versus_etapa_anterior', 'criterio']


# 02. FUNCIONES AUXILIARES

def write_csv(df, name, out):
    df.to_csv(out / name, index=False, encoding='utf-8-sig', float_format='%.6f')


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k,v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if pd.isna(obj) if np.isscalar(obj) and not isinstance(obj, (dict,list,tuple)) else False:
        return None
    return obj


def wilson_interval(successes, n, z=1.95996398454):
    if n==0:return np.nan,np.nan
    p=successes/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return 100*(c-half),100*(c+half)


def auc_ordinal(scores, observed):
    """AUC calculada por rangos con tratamiento de empates."""
    y=np.asarray(observed,dtype=bool)
    positive=int(y.sum()); negative=len(y)-positive
    if positive==0 or negative==0:return np.nan
    ranks=pd.Series(scores).rank(method='average').to_numpy(dtype='float64')
    return float((ranks[y].sum()-positive*(positive+1)/2)/(positive*negative))


# 03. PREPARACIÓN Y AUDITORÍA DE DATOS

def audit_and_prepare(path):
    raw = pd.read_csv(path, dtype={'Invoice':'string','StockCode':'string','Customer ID':'string'}, low_memory=False)
    raw.columns = raw.columns.str.strip()
    absent = set(REQUIRED) - set(raw.columns)
    if absent:
        raise ValueError(f'Columnas faltantes: {sorted(absent)}')
    raw = raw[REQUIRED].copy()
    total = len(raw)
    raw_dup = int(raw.duplicated().sum())
    missing = {k:int(raw[k].isna().sum()) for k in REQUIRED}
    d = raw.drop_duplicates().copy()
    for col in ['Invoice', 'StockCode', 'Country']:
        d[col] = d[col].astype('string').str.strip()
    d['StockCode'] = d['StockCode'].str.upper()
    d['Customer ID'] = d['Customer ID'].astype('string').str.strip().str.replace(r'\.0+$', '', regex=True)
    d['Customer ID'] = d['Customer ID'].replace({'':'', 'nan':''}).replace('', pd.NA)
    d['InvoiceDate'] = pd.to_datetime(d['InvoiceDate'], errors='coerce')
    d['Quantity'] = pd.to_numeric(d['Quantity'], errors='coerce')
    d['Price'] = pd.to_numeric(d['Price'], errors='coerce')
    d['RevenueSigned'] = d['Quantity'] * d['Price']
    d['es_cancelacion'] = d['Invoice'].str.upper().str.startswith('C').fillna(False).astype(bool)
    d['sku_mercancia'] = d['StockCode'].str.fullmatch(r'\d{5}[A-Z]{0,2}', na=False).astype(bool)
    d['referencia_ajuste'] = ~d['sku_mercancia']
    d['fecha_valida'] = d['InvoiceDate'].notna()
    d['cantidad_valida'] = d['Quantity'].notna()
    d['precio_valido'] = d['Price'].notna()
    d['identificador_factura_valido'] = d['Invoice'].notna() & d['Invoice'].ne('')
    d['base_valida'] = d[['fecha_valida','cantidad_valida','precio_valido','identificador_factura_valido']].all(axis=1)
    d['es_venta_positiva'] = d['base_valida'] & ~d['es_cancelacion'] & d['Quantity'].gt(0) & d['Price'].gt(0)
    d['es_credito_observado'] = d['base_valida'] & d['Quantity'].lt(0) & d['Price'].gt(0)
    d['es_venta_mercancia'] = d['es_venta_positiva'] & d['sku_mercancia']
    d['es_credito_mercancia'] = d['es_credito_observado'] & d['sku_mercancia']
    sales_all = d.loc[d.es_venta_positiva].copy()
    goods = d.loc[d.es_venta_mercancia].copy()
    credits = d.loc[d.es_credito_observado].copy()
    good_credits = d.loc[d.es_credito_mercancia].copy()
    for f in (sales_all, goods, credits, good_credits):
        f['Month'] = f.InvoiceDate.dt.to_period('M').astype(str)
    workflow = [
        ('Archivo original', total, 'Todas las lineas recibidas'),
        ('Sin filas completamente duplicadas', len(d), 'Eliminar duplicados identicos de las 8 columnas originales'),
        ('Con factura, fecha, cantidad y precio legibles', int(d.base_valida.sum()), 'Excluir errores de lectura en campos imprescindibles'),
        ('Ventas positivas, sin facturas C', len(sales_all), 'Solo cantidad y precio positivos; sin facturas C'),
        ('Ventas positivas de mercancia identificable', len(goods), 'Codigos SKU de cinco digitos y cero, una o dos letras'),
    ]
    waterfall=[]
    prev=None
    for name,n,rule in workflow:
        waterfall.append(dict(etapa=name, filas=n, diferencia_versus_etapa_anterior=(prev-n) if prev is not None else 0, criterio=rule))
        prev=n
    flags = [
        ('Duplicado exacto original', raw_dup),
        ('ID cliente ausente en original', missing['Customer ID']),
        ('Descripcion ausente en original', missing['Description']),
        ('Fecha ilegible tras deduplicacion',int(d.InvoiceDate.isna().sum())),
        ('Cantidad <= 0 tras deduplicacion',int(d.Quantity.le(0).sum())),
        ('Precio <= 0 tras deduplicacion',int(d.Price.le(0).sum())),
        ('Codigo de factura de cancelacion',int(d.es_cancelacion.sum())),
        ('Creditos: cantidad negativa y precio positivo',len(credits)),
        ('Venta positiva con ID cliente ausente',int(sales_all['Customer ID'].isna().sum())),
        ('Venta de mercancia con ID cliente ausente',int(goods['Customer ID'].isna().sum())),
        ('Venta positiva con SKU no mercancia',len(sales_all)-len(goods)),
        ('Venta con cantidad > 10000',int((sales_all.Quantity>10000).sum())),
    ]
    audit = {
        'archivo':path.name, 'sha256':file_hash(path), 'filas_originales':total,
        'filas_deduplicadas':len(d), 'filas_duplicadas_exactas':raw_dup,
        'inicio':d.InvoiceDate.min().strftime('%Y-%m-%d %H:%M:%S'),
        'fin':d.InvoiceDate.max().strftime('%Y-%m-%d %H:%M:%S'),
        'filas_ventas_positivas':len(sales_all), 'filas_mercancia_positivas':len(goods),
        'facturas_ventas':int(sales_all.Invoice.nunique()),
        'facturas_mercancia':int(goods.Invoice.nunique()),
        'clientes_ventas':int(sales_all['Customer ID'].nunique()),
        'clientes_mercancia':int(goods['Customer ID'].nunique()),
        'importe_bruto_total_GBP':float(sales_all.RevenueSigned.sum()),
        'importe_bruto_mercancia_GBP':float(goods.RevenueSigned.sum()),
        'importe_bruto_no_mercancia_GBP':float(sales_all.RevenueSigned.sum()-goods.RevenueSigned.sum()),
        'creditos_observados_todos_GBP':float(credits.RevenueSigned.sum()),
        'creditos_observados_mercancia_GBP':float(good_credits.RevenueSigned.sum()),
        'saldo_movimientos_todos_GBP':float(sales_all.RevenueSigned.sum()+credits.RevenueSigned.sum()),
        'saldo_movimientos_mercancia_GBP':float(goods.RevenueSigned.sum()+good_credits.RevenueSigned.sum()),
        'gasto_mercancia_identificada_GBP':float(goods.loc[goods['Customer ID'].notna(),'RevenueSigned'].sum()),
        'participacion_mercancia_sin_ID_en_bruto_pct':float(100*goods.loc[goods['Customer ID'].isna(),'RevenueSigned'].sum()/goods.RevenueSigned.sum()),
        'clientes_sin_id_original':missing['Customer ID'], 'duplicados_original':raw_dup,
        'paises_mercancia':int(goods.Country.nunique()),
        'sku_mercancia':int(goods.StockCode.nunique()),
        'porcentaje_creditos_sobre_bruto_mercancia':float(-100*good_credits.RevenueSigned.sum()/goods.RevenueSigned.sum()),
    }
    services = sales_all.loc[~sales_all.sku_mercancia].groupby('StockCode',dropna=False).agg(lineas=('Invoice','size'),importe_GBP=('RevenueSigned','sum')).sort_values('importe_GBP',ascending=False).reset_index()
    return d,sales_all,goods,credits,good_credits,audit,pd.DataFrame(waterfall),pd.DataFrame(flags,columns=['incidencia','filas']),services


def sensitivity(goods, all_sale, good_credits, baseline):
    same_raw = pd.read_csv(baseline, dtype={'Invoice':'string','StockCode':'string'},low_memory=False)
    same_raw['StockCode']=same_raw.StockCode.astype('string').str.strip().str.upper()
    same_raw['Invoice']=same_raw.Invoice.astype('string').str.strip()
    positive=same_raw.Invoice.str.upper().str.startswith('C').fillna(False).eq(False) & same_raw.Quantity.gt(0) & same_raw.Price.gt(0)
    physical=same_raw.StockCode.str.fullmatch(r'\d{5}[A-Z]{0,2}',na=False)
    raw_gross=float((same_raw.loc[positive&physical,'Quantity']*same_raw.loc[positive&physical,'Price']).sum())
    amts=goods.RevenueSigned
    threshold=float(amts.quantile(.999))
    no_top=float(amts.loc[amts.le(threshold)].sum())
    out=pd.DataFrame([
      dict(escenario='Bruto original sin quitar duplicados (mercancia)',GBP=raw_gross,criterio='No deduplicar filas exactamente iguales'),
      dict(escenario='Bruto depurado (mercancia)',GBP=float(amts.sum()),criterio='Deduplicacion exacta y codigos fisicos normalizados'),
      dict(escenario='Saldo observado (mercancia)',GBP=float(amts.sum()+good_credits.RevenueSigned.sum()),criterio='Bruto depurado menos creditos de cantidad negativa'),
      dict(escenario='Bruto depurado sin 0.1% de lineas mas grandes',GBP=no_top,criterio=f'Prueba de sensibilidad; umbral linea GBP {threshold:.2f}, NO limpieza oficial'),
      dict(escenario='Bruto depurado de todas las referencias',GBP=float(all_sale.RevenueSigned.sum()),criterio='Incluye portes y movimientos de servicios o ajustes positivos'),
    ])
    out['diferencia_vs_bruto_mercancia_GBP']=out.GBP-amts.sum()
    out['diferencia_vs_bruto_mercancia_pct']=100*out.diferencia_vs_bruto_mercancia_GBP/amts.sum()
    return out


# 04. ANÁLISIS DESCRIPTIVO Y SEGMENTACIÓN RFM

def sales_orders(goods):
    known=goods.dropna(subset=['Customer ID'])
    ambiguous=known.groupby('Invoice')['Customer ID'].nunique().gt(1).sum()
    if ambiguous:
        warnings.warn(f'{ambiguous} facturas con mas de un ID de cliente; se agrupan por pareja factura-cliente')
    orders = known.groupby(['Customer ID','Invoice'],as_index=False).agg(fecha=('InvoiceDate','min'), importe_GBP=('RevenueSigned','sum'), SKU=('StockCode','nunique'), unidades=('Quantity','sum'))
    return orders,int(ambiguous)


def score_equal_ties(series, higher_better=True):
    """Puntuación por quintiles sin separar valores empatados."""
    pct = series.rank(method='average', pct=True, ascending=higher_better)
    return np.ceil(pct.mul(5).clip(lower=1e-8)).clip(1,5).astype('int8')


def rfm_from_orders(orders, cutoff):
    hist = orders.loc[orders.fecha.lt(cutoff)].copy()
    if hist.empty:
        return pd.DataFrame()
    rfm = hist.groupby('Customer ID').agg(
        primera_observada=('fecha','min'), ultima_observada=('fecha','max'),
        Frequency=('Invoice','nunique'), Monetary=('importe_GBP','sum')
    ).reset_index()
    rfm['Recency']=(pd.Timestamp(cutoff).normalize()-rfm.ultima_observada.dt.normalize()).dt.days
    rfm['R_score'] = score_equal_ties(rfm.Recency, higher_better=False)
    rfm['F_score'] = score_equal_ties(rfm.Frequency, higher_better=True)
    rfm['M_score'] = score_equal_ties(rfm.Monetary, higher_better=True)
    rfm['RFM_score']=rfm[['R_score','F_score','M_score']].sum(axis=1)
    rfm['Segmento']=np.select([
        rfm.R_score.ge(4) & rfm.F_score.ge(4) & rfm.M_score.ge(4),
        rfm.R_score.ge(3) & rfm.F_score.ge(3),
        rfm.R_score.ge(4) & rfm.F_score.le(2),
        rfm.R_score.le(2) & rfm.F_score.ge(3),
    ], SEGMENTS[:4], default=SEGMENTS[4])
    rfm['antiguedad_observada_dias']=(pd.Timestamp(cutoff).normalize()-rfm.primera_observada.dt.normalize()).dt.days
    rfm['ticket_promedio_historico_GBP']=rfm.Monetary/rfm.Frequency
    return rfm


def rfm_group(rfm):
    grp=rfm.groupby('Segmento').agg(
        clientes=('Customer ID','nunique'),recencia_mediana_dias=('Recency','median'),
        facturas_mediana=('Frequency','median'),gasto_mediano_GBP=('Monetary','median'),
        gasto_total_GBP=('Monetary','sum'), recencia_promedio_dias=('Recency','mean')
    ).reindex(SEGMENTS).reset_index()
    grp['pct_clientes']=grp.clientes/grp.clientes.sum()*100
    grp['pct_gasto']=grp.gasto_total_GBP/grp.gasto_total_GBP.sum()*100
    return grp


def concentration(rfm):
    v=np.sort(rfm.Monetary.to_numpy(dtype='float64'))
    n=len(v); s=v.sum()
    gini=(2*np.arange(1,n+1)-n-1).dot(v)/(n*s)
    return {'gini_gasto_bruto_clientes':float(gini),
            'participacion_top_1_pct':float(v[-int(np.ceil(n*.01)):].sum()/s*100),
            'participacion_top_10_pct':float(v[-int(np.ceil(n*.10)):].sum()/s*100),
            'clientes_con_una_factura':int((rfm.Frequency==1).sum()),
            'porcentaje_clientes_con_una_factura':float(100*(rfm.Frequency==1).mean())}


def cohort_second_invoice(orders, last_date, horizon=90):
    sort=orders.sort_values(['Customer ID','fecha','Invoice']).copy()
    sort['orden']=sort.groupby('Customer ID').cumcount()+1
    first=sort.loc[sort.orden==1,['Customer ID','fecha']].rename(columns={'fecha':'primera'})
    second=sort.loc[sort.orden==2,['Customer ID','fecha']].rename(columns={'fecha':'segunda'})
    cohort=first.merge(second,on='Customer ID',how='left')
    cohort['elegible']=cohort.primera.le(last_date-pd.Timedelta(days=horizon))
    cohort['segunda_en_90d']=cohort.segunda.notna()&cohort.segunda.le(cohort.primera+pd.Timedelta(days=horizon))
    cohort['mes_primera_observada']=cohort.primera.dt.to_period('M').astype(str)
    eligible=cohort.loc[cohort.elegible].copy()
    monthly=eligible.groupby('mes_primera_observada').agg(clientes_elegibles=('Customer ID','size'),clientes_segunda_90d=('segunda_en_90d','sum')).reset_index()
    monthly['tasa_segunda_90d_pct']=100*monthly.clientes_segunda_90d/monthly.clientes_elegibles
    n=len(eligible); s=int(eligible.segunda_en_90d.sum()); low,high=wilson_interval(s,n)
    return monthly,{'n_elegibles_90d':n,'n_segunda_compra_90d':s,'tasa_segunda_90d_pct':100*s/n,'IC95_inferior_pct':low,'IC95_superior_pct':high}


def monthly_analysis(sales_all,goods,good_credits):
    gross=goods.groupby('Month').agg(ventas_brutas_mercancia_GBP=('RevenueSigned','sum'),lineas=('Invoice','size'),facturas=('Invoice','nunique')).reset_index()
    tot=sales_all.groupby('Month').RevenueSigned.sum().rename('ventas_brutas_todos_GBP').reset_index()
    credits=good_credits.groupby('Month').RevenueSigned.sum().rename('credito_mercancia_GBP').reset_index()
    monthly=gross.merge(tot,on='Month',how='left').merge(credits,on='Month',how='left').fillna({'credito_mercancia_GBP':0})
    monthly['saldo_movimientos_mercancia_GBP']=monthly.ventas_brutas_mercancia_GBP+monthly.credito_mercancia_GBP
    monthly['ticket_bruto_por_factura_mercancia_GBP']=monthly.ventas_brutas_mercancia_GBP/monthly.facturas
    monthly['anio']=monthly.Month.str[:4].astype(int)
    monthly['mes']=monthly.Month.str[-2:].astype(int)
    jan_nov=monthly.loc[monthly.anio.isin([2010,2011]) & monthly.mes.le(11)]
    pivot=jan_nov.pivot(index='mes',columns='anio',values='saldo_movimientos_mercancia_GBP').rename(columns={2010:'saldo_2010_GBP',2011:'saldo_2011_GBP'}).reset_index()
    pivot['variacion_pct']=100*(pivot.saldo_2011_GBP/pivot.saldo_2010_GBP-1)
    return monthly,pivot


def product_analysis(goods):
    prod=goods.groupby('StockCode').agg(ventas_mercancia_GBP=('RevenueSigned','sum'), unidades=('Quantity','sum'),facturas=('Invoice','nunique')).reset_index().sort_values(['ventas_mercancia_GBP','StockCode'],ascending=[False,True]).reset_index(drop=True)
    total=prod.ventas_mercancia_GBP.sum()
    prod['participacion_pct']=100*prod.ventas_mercancia_GBP/total
    prod['acumulado_pct']=prod.participacion_pct.cumsum()
    prod['acumulado_anterior_pct']=prod.acumulado_pct-prod.participacion_pct
    prod['clase_ABC']=np.select([prod.acumulado_anterior_pct.lt(80),prod.acumulado_anterior_pct.lt(95)],['A','B'],default='C')
    abc=prod.groupby('clase_ABC').agg(productos=('StockCode','size'),ingreso_bruto_GBP=('ventas_mercancia_GBP','sum')).reindex(['A','B','C']).reset_index()
    abc['pct_sku']=100*abc.productos/abc.productos.sum()
    abc['pct_ventas']=100*abc.ingreso_bruto_GBP/abc.ingreso_bruto_GBP.sum()
    return prod,abc


def pair_analysis(goods, n_top=75, min_common=70):
    from collections import Counter
    from itertools import combinations
    candidates=set(goods.groupby('StockCode').Invoice.nunique().nlargest(n_top).index)
    baskets=goods[['Invoice','StockCode']].drop_duplicates()
    full_orders=int(baskets.Invoice.nunique())
    freq=baskets.loc[baskets.StockCode.isin(candidates)].groupby('StockCode').Invoice.nunique().to_dict()
    limited=baskets.loc[baskets.StockCode.isin(candidates)]
    cnt=Counter()
    for _,grp in limited.groupby('Invoice',sort=False):
        sku=sorted(grp.StockCode.unique().tolist())
        for a,b in combinations(sku,2):
            cnt[(a,b)]+=1
    pairs=[]
    for (a,b),joint in cnt.items():
        if joint<min_common:continue
        lift=joint*full_orders/(freq[a]*freq[b])
        pairs.append(dict(sku_A=a,sku_B=b,facturas_conjuntas=joint,
            soporte_pct=100*joint/full_orders,
            confianza_A_a_B_pct=100*joint/freq[a],
            confianza_B_a_A_pct=100*joint/freq[b],
            lift=lift))
    return pd.DataFrame(pairs).sort_values(['lift','facturas_conjuntas'],ascending=False).reset_index(drop=True) if pairs else pd.DataFrame(columns=['sku_A','sku_B','facturas_conjuntas','soporte_pct','confianza_A_a_B_pct','confianza_B_a_A_pct','lift'])


# 05. VALIDACIÓN TEMPORAL DE LA RECOMPRA

def temporal_validation(orders, cuts, horizon=90):
    last=orders.fecha.max()
    frame=[]; details=[]; auc_rows=[]
    for cut_str in cuts:
        cut=pd.Timestamp(cut_str)
        if last < cut+pd.Timedelta(days=horizon):
            raise ValueError(f'Corte sin 90 dias de seguimiento: {cut_str}')
        base=rfm_from_orders(orders, cut)
        future=orders.loc[orders.fecha.ge(cut)&orders.fecha.lt(cut+pd.Timedelta(days=horizon))]
        agg=future.groupby('Customer ID').agg(facturas_futuras_90d=('Invoice','nunique'),gasto_futuro_90d_GBP=('importe_GBP','sum')).reset_index()
        base=base.merge(agg, how='left',on='Customer ID')
        base['facturas_futuras_90d']=base.facturas_futuras_90d.fillna(0).astype('int64')
        base['gasto_futuro_90d_GBP']=base.gasto_futuro_90d_GBP.fillna(0.)
        base['recompra_futura_90d']=base.facturas_futuras_90d.gt(0)
        base['corte']=cut_str
        auc_rows.append({'corte':cut_str,'n_clientes':len(base),'tasa_base_pct':float(100*base.recompra_futura_90d.mean()),
                         'AUC_recencia':auc_ordinal(base.R_score,base.recompra_futura_90d),
                         'AUC_frecuencia':auc_ordinal(base.F_score,base.recompra_futura_90d),
                         'AUC_monetario':auc_ordinal(base.M_score,base.recompra_futura_90d),
                         'AUC_recencia_mas_frecuencia':auc_ordinal(base.R_score+base.F_score,base.recompra_futura_90d),
                         'AUC_RFM':auc_ordinal(base.RFM_score,base.recompra_futura_90d)})
        details.append(base)
        for seg,d in base.groupby('Segmento'):
            n=len(d); hits=int(d.recompra_futura_90d.sum())
            lower,upper=wilson_interval(hits,n)
            frame.append(dict(corte=cut_str,segmento=seg,clientes=n,clientes_con_compra_90d=hits,
               tasa_recompra_pct=100*hits/n,IC95_inferior_pct=lower,IC95_superior_pct=upper,
               gasto_posterior_90d_GBP=float(d.gasto_futuro_90d_GBP.sum()),
               gasto_futuro_por_cliente_GBP=float(d.gasto_futuro_90d_GBP.mean()),
               gasto_historico_GBP=float(d.Monetary.sum()),
               porcentaje_gasto_futuro_del_corte=100*d.gasto_futuro_90d_GBP.sum()/max(1,base.gasto_futuro_90d_GBP.sum())))
    all_results=pd.concat(details,ignore_index=True)
    summary=pd.DataFrame(frame).sort_values(['corte','segmento'])
    comparison=[]
    for cut,sub in summary.groupby('corte'):
        top=sub.set_index('segmento')
        target=top.loc['Alto valor']
        risk=top.loc['En riesgo']
        universe=all_results.loc[all_results.corte==cut]
        base_rate=100*universe.recompra_futura_90d.mean()
        comparison.append(dict(corte=cut,clientes=int(len(universe)),tasa_base_pct=base_rate,
               tasa_alto_valor_pct=float(target.tasa_recompra_pct), tasa_en_riesgo_pct=float(risk.tasa_recompra_pct),
               diferencia_alto_menos_riesgo_pp=float(target.tasa_recompra_pct-risk.tasa_recompra_pct),
               razon_alto_vs_riesgo=(float(target.tasa_recompra_pct/risk.tasa_recompra_pct) if risk.tasa_recompra_pct else np.nan),
               participacion_gasto_futuro_alto_valor_pct=float(target.porcentaje_gasto_futuro_del_corte)))
    return all_results,summary,pd.DataFrame(comparison),pd.DataFrame(auc_rows)


def bootstrap_auc_last_cut(historical, n_boot=500, seed=20261121):
    """Bootstrap pareado y estratificado del último corte."""
    last=historical.corte.max()
    d=historical.loc[historical.corte.eq(last)]
    y=d.recompra_futura_90d.to_numpy(dtype=bool)
    ix_yes=np.flatnonzero(y); ix_no=np.flatnonzero(~y)
    total=(d.R_score+d.F_score).to_numpy(dtype=float)
    full=d.RFM_score.to_numpy(dtype=float)
    rng=np.random.default_rng(seed)
    samples=[]
    for _ in range(n_boot):
        idx=np.r_[rng.choice(ix_yes,len(ix_yes),replace=True), rng.choice(ix_no,len(ix_no),replace=True)]
        a=auc_ordinal(full[idx],y[idx]);b=auc_ordinal(total[idx],y[idx])
        samples.append((a,b,a-b))
    vals=np.array(samples)
    labels=['AUC_RFM','AUC_R_mas_F','Diferencia_AUC_RFM_menos_RF']
    return pd.DataFrame([dict(corte=last,indicador=l,n_boot=n_boot,semilla=seed,
              estimacion=float([auc_ordinal(full,y),auc_ordinal(total,y),auc_ordinal(full,y)-auc_ordinal(total,y)][i]),
              bootstrap_pct_2_5=float(np.percentile(vals[:,i],2.5)),
              bootstrap_pct_97_5=float(np.percentile(vals[:,i],97.5))) for i,l in enumerate(labels)])


# 06. MODELOS DE APRENDIZAJE AUTOMÁTICO

def customer_features_at_cut(orders, cut):
    """Atributos calculados con datos anteriores al corte."""
    cut = pd.Timestamp(cut)
    hist = orders.loc[orders.fecha.lt(cut)].sort_values(['Customer ID','fecha','Invoice']).copy()
    x = hist.groupby('Customer ID').agg(
        sku_promedio_por_factura=('SKU','mean'),
        unidades_promedio_por_factura=('unidades','mean'),
        ingreso_medio_por_factura=('importe_GBP','mean')
    ).reset_index()
    recent = hist.loc[hist.fecha.ge(cut-pd.Timedelta(days=90))].groupby('Customer ID').agg(
        facturas_ultimos_90d=('Invoice','nunique'),
        importe_ultimos_90d_GBP=('importe_GBP','sum')
    ).reset_index()
    hist['intervalo_dias'] = hist.groupby('Customer ID').fecha.diff().dt.total_seconds()/86400
    gaps = hist.groupby('Customer ID').intervalo_dias.mean().rename('intervalo_medio_entre_facturas_dias').reset_index()
    x=x.merge(recent,how='left',on='Customer ID').merge(gaps,how='left',on='Customer ID')
    for col in ['facturas_ultimos_90d','importe_ultimos_90d_GBP','intervalo_medio_entre_facturas_dias']:
        x[col]=x[col].fillna(0)
    x['corte']=cut.strftime('%Y-%m-%d')
    return x


def classification_metrics(y, p, top_fraction=.20):
    """La columna AUC_PR corresponde a la precisión promedio (AP)."""
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    y=np.asarray(y,dtype=int)
    p=np.asarray(p,dtype=float)
    pred=(p>=.5).astype(int)
    n=max(1,int(np.ceil(len(p)*top_fraction)))
    limit=float(np.sort(p)[-n])
    above=p>limit
    tied=p==limit
    remaining=n-int(above.sum())
    expected_hits=float(y[above].sum())+float(y[tied].sum())*remaining/max(1,int(tied.sum()))
    base=float(y.mean())
    precision_top=float(expected_hits/n)
    return dict(n_clientes=len(y), tasa_real_recompra_pct=100*base,
       AUC_ROC=float(roc_auc_score(y,p)), AUC_PR=float(average_precision_score(y,p)),
       Brier=float(brier_score_loss(y,p)),precision_umbral_050=float(precision_score(y,pred,zero_division=0)),
       recall_umbral_050=float(recall_score(y,pred,zero_division=0)),
       F1_umbral_050=float(f1_score(y,pred,zero_division=0)),
       falsos_positivos=int(confusion_matrix(y,pred)[0,1]),
       precision_top20_pct=100*precision_top,
       lift_top20=float(precision_top/base) if base else float('nan'))


def run_customer_ml(orders, historical, out):
    """Entrenamiento, validación y prueba en cortes temporales separados."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer,StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score

    features=pd.concat([customer_features_at_cut(orders,c) for c in CUTS],ignore_index=True)
    allrows=historical.merge(features,on=['Customer ID','corte'],how='left',validate='one_to_one')
    columns=['Recency','Frequency','Monetary','antiguedad_observada_dias',
            'ingreso_medio_por_factura','sku_promedio_por_factura','unidades_promedio_por_factura',
            'facturas_ultimos_90d','importe_ultimos_90d_GBP','intervalo_medio_entre_facturas_dias']
    assert allrows[columns].notna().all().all(), 'Hay filas sin caracteristicas anteriores al corte'
    train=allrows.loc[allrows.corte.isin(CUTS[:4])].copy()
    val=allrows.loc[allrows.corte.eq(CUTS[4])].copy()
    test=allrows.loc[allrows.corte.eq(CUTS[5])].copy()
    assert all(pd.Timestamp(c)+pd.Timedelta(days=90)<=pd.Timestamp(CUTS[4]) for c in CUTS[:4])
    assert pd.Timestamp(CUTS[4])+pd.Timedelta(days=90)<=pd.Timestamp(CUTS[5])
    X_train=train[columns].astype(float);X_val=val[columns].astype(float);X_test=test[columns].astype(float)
    y_train=train.recompra_futura_90d.astype(int)
    y_val=val.recompra_futura_90d.astype(int)
    y_test=test.recompra_futura_90d.astype(int)
    logistic=Pipeline([
        ('imputar',SimpleImputer(strategy='median')),
        ('log1p',FunctionTransformer(np.log1p,validate=False)),
        ('escalar',StandardScaler()),
        ('modelo',LogisticRegression(max_iter=2000,random_state=2026))])
    forest=Pipeline([
        ('imputar',SimpleImputer(strategy='median')),
        ('modelo',RandomForestClassifier(n_estimators=180,max_depth=9,min_samples_leaf=25,
                        max_features='sqrt',n_jobs=-1,random_state=2026))])
    booster=Pipeline([
        ('imputar',SimpleImputer(strategy='median')),
        ('modelo',HistGradientBoostingClassifier(max_iter=140,learning_rate=.05,max_leaf_nodes=15,
                        min_samples_leaf=40,l2_regularization=2.0,random_state=2026))])
    models={'Regresion logistica':logistic,'Random Forest':forest,'Gradient Boosting':booster}
    metrics=[]; predictions={}
    for name,model in models.items():
        model.fit(X_train,y_train)
        pv=model.predict_proba(X_val)[:,1]
        pt=model.predict_proba(X_test)[:,1]
        predictions[name] = pt
        for tag,frame,y,p in [('Validacion',val,y_val,pv),('Prueba',test,y_test,pt)]:
            metrics.append(dict(modelo=name,periodo=tag,corte=frame.corte.iloc[0],**classification_metrics(y,p)))
    for name,scorecol in [('RFM base','RFM_score'),('Recencia + frecuencia','RF_score')]:
        train_score = train.RFM_score if scorecol=='RFM_score' else train.R_score+train.F_score
        fit_iso=IsotonicRegression(out_of_bounds='clip',y_min=0,y_max=1)
        fit_iso.fit(train_score.to_numpy(dtype=float),y_train.to_numpy())
        for tag,frame,y in [('Validacion',val,y_val),('Prueba',test,y_test)]:
            score=frame.RFM_score if scorecol=='RFM_score' else frame.R_score+frame.F_score
            p=fit_iso.predict(score.to_numpy(dtype=float))
            if tag=='Prueba':predictions[name]=p
            metrics.append(dict(modelo=name,periodo=tag,corte=frame.corte.iloc[0],**classification_metrics(y,p)))
    metr=pd.DataFrame(metrics)
    rank=metr.loc[metr.periodo.eq('Validacion')].sort_values(['AUC_ROC','AUC_PR','modelo'],ascending=[False,False,True])
    chosen=rank.iloc[0].modelo
    test_export=test[['Customer ID','corte','Segmento','Recency','Frequency','Monetary',
                      'recompra_futura_90d']].copy()
    for name,prob in predictions.items():
        test_export['prob_'+name.lower().replace(' ','_').replace('+','mas')]=prob
    test_export['prob_modelo_seleccionado']=predictions[chosen]
    sorted_idx=test_export.sort_values(['prob_modelo_seleccionado','Customer ID'],
                    ascending=[False,True],kind='stable').index[:int(np.ceil(len(test_export)*.2))]
    test_export['prioridad_top20']=test_export.index.isin(sorted_idx)
    rng=np.random.default_rng(20261121)
    y=y_test.to_numpy()
    p_new=np.asarray(predictions[chosen]);p_base=np.asarray(predictions['RFM base'])
    pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
    delta=[]
    for _ in range(500):
        idx=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
        delta.append(roc_auc_score(y[idx],p_new[idx])-roc_auc_score(y[idx],p_base[idx]))
    diff=dict(modelo_seleccionado=chosen, corte_prueba=CUTS[5],
        diferencia_AUC_seleccionado_menos_RFM=float(roc_auc_score(y,p_new)-roc_auc_score(y,p_base)),
        IC95_bootstrap_inferior=float(np.percentile(delta,2.5)),
        IC95_bootstrap_superior=float(np.percentile(delta,97.5)),
        n_bootstrap=500, nota='IC condicional al corte y clientes; no valida otros negocios ni cambios de mercado')
    return metr,test_export,chosen,diff,columns


# 07. PRONÓSTICO DE VENTAS SEMANALES

def build_weekly_sales(goods):
    frame=goods[['InvoiceDate','RevenueSigned']].copy()
    frame['semana']=frame.InvoiceDate.dt.to_period('W-SUN').dt.start_time
    week=frame.groupby('semana').RevenueSigned.sum().sort_index()
    weeks=pd.date_range(week.index.min(),week.index.max(),freq='W-MON')
    week=week.reindex(weeks,fill_value=0.0)
    origin=goods.InvoiceDate.min().normalize();end=goods.InvoiceDate.max().normalize()
    week=week.loc[(week.index>=origin)&(week.index+pd.Timedelta(days=6)<=end)]
    assert len(week)>65 and (week>=0).all()
    return week


def forecasting_features(history_values, date):
    values=np.asarray(history_values,dtype=float)
    week_of_year=float(pd.Timestamp(date).isocalendar().week)
    w=2*np.pi*week_of_year/52.18
    return np.array([values[-1],values[-2],values[-4],values[-8],values[-12],
            np.mean(values[-4:]),np.mean(values[-12:]),np.sin(w),np.cos(w)],dtype=float)


def forecast_ml_weekly(history, n=4):
    from sklearn.ensemble import RandomForestRegressor
    dates=history.index
    y=history.to_numpy(dtype=float)
    X=[];target=[]
    for i in range(12,len(y)):
        X.append(forecasting_features(y[:i],dates[i]));target.append(y[i])
    model=RandomForestRegressor(n_estimators=140,max_depth=7,min_samples_leaf=3,
          max_features=0.85,random_state=20261121,n_jobs=-1)
    model.fit(np.asarray(X),np.asarray(target))
    pred=[]
    values=y.tolist()
    for j in range(n):
        future_date=dates[-1]+pd.Timedelta(weeks=j+1)
        x=forecasting_features(values,future_date).reshape(1,-1)
        z=max(0.0,float(model.predict(x)[0]))
        pred.append(z);values.append(z)
    return np.asarray(pred)


def forecast_evaluation(goods):
    series=build_weekly_sales(goods)
    starts=['2011-02-07','2011-04-04','2011-06-06','2011-08-01','2011-10-03']
    records=[];actual_preds=[]
    for start in starts:
        cut=pd.Timestamp(start)
        hist=series.loc[series.index < cut]
        actual=series.loc[series.index >= cut].head(4)
        assert len(actual)==4 and len(hist)>52, f'Semana de evaluacion incompleta {start}'
        alternatives={
            'Ultima semana':np.repeat(hist.iloc[-1],4).astype(float),
            'Promedio 4 semanas':np.repeat(hist.iloc[-4:].mean(),4),
            'Misma semana ano anterior':series.reindex(actual.index-pd.Timedelta(weeks=52)).to_numpy(dtype=float),
            'Random Forest temporal':forecast_ml_weekly(hist,4),
        }
        for method,pred in alternatives.items():
            assert np.isfinite(pred).all() and (pred>=0).all()
            err=np.abs(actual.to_numpy()-pred)
            records.append(dict(corte=start,modelo=method,semanas=4,
                  MAE_GBP=float(err.mean()),WAPE_pct=float(100*err.sum()/max(1,float(actual.sum()))),
                  sesgo_GBP=float(np.mean(pred-actual.to_numpy()))))
            for week,yhat,ytrue in zip(actual.index,pred,actual):
                actual_preds.append(dict(corte=start,modelo=method,semana=str(week.date()),
                          venta_real_GBP=float(ytrue),prediccion_GBP=float(yhat),
                          error_absoluto_GBP=abs(float(ytrue-yhat))))
    results=pd.DataFrame(records)
    summary=results.groupby('modelo').agg(MAE_promedio_GBP=('MAE_GBP','mean'),
               WAPE_promedio_pct=('WAPE_pct','mean'),cortes=('corte','nunique')).reset_index()
    best=summary.sort_values(['WAPE_promedio_pct','MAE_promedio_GBP']).iloc[0].modelo
    return series,results,pd.DataFrame(actual_preds),summary,best


# 08. ESCENARIOS ECONÓMICOS Y DECISIONES COMERCIALES

def economic_scenarios(orders, client_test, winner):
    hist=orders.loc[orders.fecha.lt(pd.Timestamp(CUTS[-1]))]
    ticket=float(hist.importe_GBP.median())
    n=int(client_test.prioridad_top20.sum())
    scenarios=[]
    for margin in [.20,.35,.50]:
        for cost in [.5,1.,2.]:
            for uplift_pp in [1.,3.,5.]:
                increase=n*uplift_pp/100.
                revenue=increase*ticket
                gross_profit=revenue*margin
                total_cost=n*cost
                scenarios.append(dict(
                    grupo='20% de clientes con mayor puntuacion en prueba',modelo=winner,
                    n_clientes_objetivo=n, ticket_mediano_historico_GBP=ticket,
                    supuesto_margen_contribucion_pct=margin*100,
                    supuesto_costo_contacto_GBP=cost,
                    supuesto_aumento_absoluto_recompra_pp=uplift_pp,
                    nuevas_compras_esperadas_hipoteticas=increase,
                    facturacion_incremental_hipotetica_GBP=revenue,
                    contribucion_incremental_menos_contactos_GBP=gross_profit-total_cost,
                    aumento_pp_equilibrio=(100*total_cost/(n*ticket*margin)) if ticket*margin else float('nan'),
                    advertencia='SUPUESTOS. No existe evidencia de efecto de campana ni costos o margen reales'))
    return pd.DataFrame(scenarios)


def decisions(audit, comparison, seg, abc, conf, risk):
    last = comparison.sort_values('corte').iloc[-1]
    high_risk = risk.loc[(risk.Segmento=='En riesgo') & risk.M_score.ge(4)]
    low = risk.loc[risk.Segmento.eq('En riesgo')]
    top_abc=abc.loc[abc.clase_ABC.eq('A')].iloc[0]
    return pd.DataFrame([
      dict(decision='Revisar registro y devoluciones antes de presupuestar', evidencia=f"Los creditos reducen en {audit['porcentaje_creditos_sobre_bruto_mercancia']:.2f}% el bruto positivo de mercancia",propuesta='Panel mensual con bruto, devoluciones y saldo; conciliar facturas',indicador_ex_post='Diferencia contra contabilidad y porcentaje de devoluciones',condicion='Se requiere conciliacion real con notas de credito y pagos'),
      dict(decision='Priorizar fidelizacion o reactivacion segun recompra futura observada', evidencia=f"Corte {last.corte}: alto valor {last.tasa_alto_valor_pct:.1f}% vs en riesgo {last.tasa_en_riesgo_pct:.1f}% a 90d",propuesta='Prueba controlada de comunicacion con grupos comparables',indicador_ex_post='Cambio en recompra a 90 dias y margen incremental',condicion='La diferencia historica no representa el efecto causal de la campana'),
      dict(decision='Contactar primero clientes de mayor gasto historico dentro de inactivos',evidencia=f'{len(high_risk)} clientes de mayor puntaje monetario de {len(low)} en riesgo al cierre',propuesta='Piloto de contacto limitado y medible; presupuesto con tope por contacto',indicador_ex_post='Conversion de contactos y beneficio incremental',condicion='No afirmar que todo el gasto historico es recuperable'),
      dict(decision='Monitorear disponibilidad del catalogo principal',evidencia=f"{int(top_abc.productos)} SKU de tipo A representan {top_abc.pct_ventas:.1f}% del bruto de mercancia",propuesta='Control semanal de roturas para referencias A',indicador_ex_post='Dias sin existencias, demanda perdida y margen',condicion='Sin inventario ni costes no se calcula stock optimo'),
      dict(decision='Explorar venta cruzada con pares recurrentes',evidencia=f"Pareja {conf.iloc[0].sku_A} y {conf.iloc[0].sku_B}, lift {conf.iloc[0].lift:.2f} y {int(conf.iloc[0].facturas_conjuntas)} facturas" if len(conf) else 'Sin parejas con soporte suficiente',propuesta='Mostrar sugerencia a muestra aleatoria y comparar con control',indicador_ex_post='Tasa de adicion, gasto y margen incremental',condicion='Las asociaciones no demuestran causalidad; revisar productos y disponibilidad'),
    ])


# 09. GRÁFICAS

def graph_monthly(monthly, out):
    fig,ax=plt.subplots(figsize=(11,4.5))
    ax.plot(monthly.Month,monthly.ventas_brutas_mercancia_GBP/1e6,marker='.',label='Ventas positivas brutas')
    ax.plot(monthly.Month,monthly.saldo_movimientos_mercancia_GBP/1e6,marker='.',label='Saldo con creditos registrados')
    ax.set_ylabel('Millones de GBP');ax.set_title('Mercancia: ventas y movimientos con signo')
    ax.tick_params(axis='x',rotation=75);ax.legend();ax.grid(alpha=.18);fig.tight_layout()
    fig.savefig(out/'fig_01_ventas_y_creditos.png',dpi=180);plt.close(fig)


def graph_segment(segments,out):
    s=segments.set_index('Segmento').reindex(SEGMENTS)
    x=np.arange(len(s));fig,ax=plt.subplots(figsize=(10,4.5))
    ax.bar(x-.2,s.pct_clientes,width=.4,label='Clientes (%)')
    ax.bar(x+.2,s.pct_gasto,width=.4,label='Gasto historico (%)')
    ax.set_xticks(x,s.index,rotation=15,ha='right');ax.set_ylabel('Participacion (%)')
    ax.set_title('Concentracion por segmento RFM, ultimo corte')
    ax.legend();fig.tight_layout();fig.savefig(out/'fig_04_concentracion_segmentos.png',dpi=180);plt.close(fig)


def graph_validation(temporal, out):
    piv=temporal.pivot(index='corte',columns='segmento',values='tasa_recompra_pct').reindex(columns=SEGMENTS)
    fig,ax=plt.subplots(figsize=(10,5))
    for c in piv.columns:ax.plot(piv.index,piv[c],marker='o',label=c)
    ax.set_ylabel('Clientes que vuelven a comprar a 90 dias (%)')
    ax.set_xlabel('Corte historico de evaluacion');ax.set_title('Validacion retrospectiva con datos futuros separados')
    ax.set_ylim(0,100);ax.grid(alpha=.2);ax.legend(loc='best',fontsize=8)
    fig.tight_layout();fig.savefig(out/'fig_02_validacion_recompra.png',dpi=180);plt.close(fig)


def graph_final_validation(results,out):
    last=results.corte.max()
    sub=results.loc[results.corte==last].set_index('segmento').reindex(SEGMENTS)
    fig,ax=plt.subplots(figsize=(10,4.5))
    ys=np.arange(len(sub))
    ax.barh(ys,sub.tasa_recompra_pct)
    errors=np.vstack([sub.tasa_recompra_pct-sub.IC95_inferior_pct,sub.IC95_superior_pct-sub.tasa_recompra_pct])
    ax.errorbar(sub.tasa_recompra_pct,ys,xerr=errors,fmt='none',ecolor='black',capsize=3)
    ax.set_yticks(ys,sub.index);ax.invert_yaxis();ax.set_xlim(0,min(100,sub.IC95_superior_pct.max()+12))
    ax.set_xlabel('Recompra de mercancia a 90 dias (%)')
    ax.set_title('Corte %s: barras de incertidumbre Wilson 95%%'%last)
    ax.grid(axis='x',alpha=.18);fig.tight_layout();fig.savefig(out/'fig_03_ultimo_corte_IC95.png',dpi=180);plt.close(fig)


def graph_auc(auc,out):
    fig,ax=plt.subplots(figsize=(10,4.5))
    for col in ['AUC_recencia','AUC_frecuencia','AUC_monetario','AUC_recencia_mas_frecuencia','AUC_RFM']:
        ax.plot(auc.corte,auc[col],marker='o',label=col.replace('AUC_','').replace('_',' '))
    ax.axhline(.5,linestyle='--',linewidth=1,color='grey',label='Aleatorio (0.50)')
    ax.set_ylabel('AUC de recompra a 90 dias')
    ax.set_ylim(.45,1);ax.set_title('Comparacion prospectiva de indicadores ordinales')
    ax.tick_params(axis='x',rotation=15);ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.18)
    fig.tight_layout();fig.savefig(out/'fig_07_AUC_indicadores.png',dpi=180);plt.close(fig)


def graph_abc(abc,out):
    fig,ax=plt.subplots(figsize=(7,4))
    ax.bar(abc.clase_ABC,abc.pct_ventas)
    ax.set_ylabel('Ventas positivas de mercancia (%)');ax.set_title('Analisis ABC por SKU')
    for i,r in abc.iterrows():ax.text(i,r.pct_ventas+1,f'{r.pct_ventas:.1f}%',ha='center')
    ax.set_ylim(0,100);fig.tight_layout();fig.savefig(out/'fig_05_abc_sku.png',dpi=180);plt.close(fig)


def graph_cohort(monthly,out):
    x=np.arange(len(monthly));fig,ax=plt.subplots(figsize=(10,4))
    ax.plot(x,monthly.tasa_segunda_90d_pct,marker='o')
    ax.set_xticks(x,monthly.mes_primera_observada,rotation=70)
    ax.set_ylabel('Segunda factura antes de 90 dias (%)');ax.set_title('Cohortes con ventana completa de observacion')
    ax.grid(alpha=.18);fig.tight_layout();fig.savefig(out/'fig_06_cohortes_segunda_compra.png',dpi=180);plt.close(fig)


def plot_ml_results(metrics, out):
    last=metrics.loc[metrics.periodo.eq('Prueba')].sort_values('AUC_ROC')
    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.barh(last.modelo,last.AUC_ROC)
    ax.set_xlim(.45,max(.90,float(last.AUC_ROC.max())+.06))
    ax.set_xlabel('AUC-ROC (prueba septiembre 2011)')
    for i,v in enumerate(last.AUC_ROC):ax.text(v+.008,i,f'{v:.3f}',va='center')
    ax.set_title('Modelos predictivos frente a indicadores RFM')
    fig.tight_layout();fig.savefig(out/'fig_08_ML_comparacion_prueba.png',dpi=180);plt.close(fig)


def plot_forecast_results(summary,out):
    s=summary.sort_values('WAPE_promedio_pct')
    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.barh(s.modelo,s.WAPE_promedio_pct)
    ax.invert_yaxis();ax.set_xlabel('WAPE medio por corte (%) — menor es mejor')
    ax.set_title('Pronostico de ventas: 5 cortes de 4 semanas')
    fig.tight_layout();fig.savefig(out/'fig_09_pronosticos_benchmark.png',dpi=180);plt.close(fig)


def plot_scenarios(scenarios,out):
    select=scenarios.loc[scenarios.supuesto_margen_contribucion_pct.eq(35)]
    pivot=select.pivot(index='supuesto_costo_contacto_GBP',columns='supuesto_aumento_absoluto_recompra_pp',
            values='contribucion_incremental_menos_contactos_GBP')
    fig,ax=plt.subplots(figsize=(6.8,4))
    colors=ax.imshow(pivot.to_numpy(),cmap='RdYlGn',aspect='auto')
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j,i,f'{pivot.iloc[i,j]:,.0f}',ha='center',va='center',fontsize=10)
    ax.set_xticks(range(len(pivot.columns)),[f'{x:.0f} pp' for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)),[f'GBP {x:.2f}' for x in pivot.index])
    ax.set_xlabel('Aumento supuesto de recompra')
    ax.set_ylabel('Costo supuesto por contacto')
    ax.set_title('Contribucion hipotetica (margen supuesto 35%)')
    fig.colorbar(colors,ax=ax,label='GBP — no observado')
    fig.tight_layout();fig.savefig(out/'fig_10_escenarios_hipoteticos.png',dpi=180);plt.close(fig)


# 10. EJECUCIÓN Y EXPORTACIÓN DE RESULTADOS

def main():
    base_dir = Path(__file__).resolve().parent
    
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=base_dir / 'data' / 'online_retail_II.csv')
    ap.add_argument('--output',type=Path,default=base_dir / 'results')
    
    
    args=ap.parse_args()
    tables_dir = args.output / 'tables'
    figures_dir = args.output / 'figures'
    metrics_dir = args.output / 'metrics'

    for directory in (tables_dir, figures_dir, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print('\n' + '=' * 72)
    print('ANALÍTICA DE DATOS Y APRENDIZAJE AUTOMÁTICO')
    print('=' * 72)
    print(f'Archivo de entrada: {args.input}')
    print(f'Directorio de salida: {args.output.resolve()}')

    # Preparación de datos
    data,sales,goods,credits,goods_credits,audit,waterfall,flags,services=audit_and_prepare(args.input)

    # Análisis descriptivo y RFM
    orders,ambiguous=sales_orders(goods)
    cutoff=goods.InvoiceDate.max()+pd.Timedelta(microseconds=1)
    rfm=rfm_from_orders(orders,cutoff)
    assert len(rfm)==int(goods['Customer ID'].nunique()),'No todos los clientes han entrado en RFM'
    assert (rfm.groupby('Frequency').F_score.nunique()<=1).all(), 'Mismo F no puede recibir dos puntuaciones'
    assert abs(goods.RevenueSigned.sum()+goods_credits.RevenueSigned.sum()-audit['saldo_movimientos_mercancia_GBP'])<0.01
    segments=rfm_group(rfm)
    concentration_metrics=concentration(rfm)
    cohort,cohort_metrics=cohort_second_invoice(orders,goods.InvoiceDate.max())

    monthly,yoy=monthly_analysis(sales,goods,goods_credits)
    prods,abc=product_analysis(goods)
    pairs=pair_analysis(goods)
    sen=sensitivity(goods,sales,goods_credits,args.input)
    risk=rfm.loc[rfm.Segmento=='En riesgo']
    priority=rfm.loc[rfm.Segmento=='En riesgo'].assign(alto_valor_historico=lambda x:x.M_score>=4).groupby('alto_valor_historico').agg(clientes=('Customer ID','size'), gasto_historico_GBP=('Monetary','sum')).reset_index()
    priority['porcentaje_del_gasto_en_riesgo']=100*priority.gasto_historico_GBP/priority.gasto_historico_GBP.sum()
    countries=goods.groupby('Country').agg(bruto_mercancia_GBP=('RevenueSigned','sum'),facturas=('Invoice','nunique')).sort_values('bruto_mercancia_GBP',ascending=False).reset_index()

    # Validación temporal
    historical, temporal, compare, auc=temporal_validation(orders,CUTS)
    interval_auc=bootstrap_auc_last_cut(historical)

    # Modelos predictivos
    ml_metrics, ml_predictions, selected_model, auc_difference, ml_features=run_customer_ml(orders,historical,args.output)

    # Pronóstico de ventas
    weekly_series,forecast_by_cut,forecast_detail,forecast_summary,best_forecast=forecast_evaluation(goods)

    # Escenarios económicos y decisiones
    scenario_table=economic_scenarios(orders,ml_predictions,selected_model)
    decisions_out=decisions(audit,compare,segments,abc,pairs,rfm)

    # Indicadores consolidados
    agg_canon={'auditoria':audit,'concentracion':concentration_metrics,'segunda_compra':cohort_metrics,'facturas_ambiguedad_ID':ambiguous,
        'corte_final':cutoff.isoformat(),'cortes_validacion':CUTS,'horizonte_dias':90,
        'seg_ultimo_corte':segments.to_dict(orient='records'),
        'comparacion_validacion':compare.to_dict(orient='records'),'auc_validacion':auc.to_dict(orient='records'), 'intervalos_auc':interval_auc.to_dict(orient='records'),
        'ABC':abc.to_dict(orient='records'),
        'prioridad_riesgo':priority.to_dict(orient='records')}
    agg_canon['machine_learning']={'mejor_modelo_segun_validacion':selected_model,
        'metricas':ml_metrics.to_dict(orient='records'),'diferencia_AUC_vs_RFM_prueba':auc_difference,
        'atributos':ml_features,'seleccion_solo_validacion':CUTS[4],'prueba_no_usada_para_seleccion':CUTS[5]}
    agg_canon['pronostico_ventas']={'mejor_metodo_por_WAPE_medio':best_forecast,
        'metricas':forecast_summary.to_dict(orient='records'),
        'criterio':'Ventas brutas semanales de mercancia. Se excluyen semanas incompletas.'}
    agg_canon['escenarios_hipoteticos']={'ticket_mediano_observado_GBP':float(scenario_table.ticket_mediano_historico_GBP.iloc[0]),
        'n_clientes_objetivo':int(scenario_table.n_clientes_objetivo.iloc[0]),
        'nota':'Margen, costo por contacto y uplift NO observados: supuestos de simulacion'}

    # Exportación de tablas
    for df,name in [
        (waterfall,'01_flujo_limpieza.csv'),(flags,'02_incidencias_calidad.csv'),(services,'03_codigos_no_mercancia.csv'),
        (sen,'04_sensibilidad_calculos.csv'),(monthly,'05_ventas_mensuales.csv'),(yoy,'06_comparacion_interanual.csv'),
        (segments,'07_segmentos_RFM_cierre.csv'),(rfm,'08_clientes_RFM_cierre.csv'),
        (priority,'09_prioridad_inactivos.csv'),(cohort,'10_cohortes_90_dias.csv'),
        (temporal,'11_validacion_cortes_90_dias.csv'),(compare,'12_comparacion_cortes.csv'),
        (historical,'13_detalle_validacion_clientes.csv'),(auc,'19_AUC_comparacion_indicadores.csv'),(interval_auc,'20_bootstrap_auc_ultimo_corte.csv'),(prods,'14_productos_ABC.csv'),
        (abc,'15_resumen_ABC.csv'),(pairs,'16_pares_productos.csv'),
        (countries,'17_paises.csv'),(decisions_out,'18_matriz_decisiones.csv')]:
        write_csv(df,name,tables_dir)
    write_csv(ml_metrics,'21_metricas_ML_validacion_prueba.csv',tables_dir)
    write_csv(ml_predictions,'22_predicciones_clientes_prueba.csv',tables_dir)
    write_csv(pd.DataFrame([auc_difference]),'23_bootstrap_AUC_ML_vs_RFM.csv',tables_dir)
    write_csv(forecast_by_cut,'24_pronosticos_por_corte.csv',tables_dir)
    write_csv(forecast_detail,'25_pronosticos_detalle_semanal.csv',tables_dir)
    write_csv(forecast_summary,'26_comparacion_pronosticos.csv',tables_dir)
    write_csv(scenario_table,'27_escenarios_hipoteticos_costos.csv',tables_dir)
    write_csv(weekly_series.rename('ventas_brutas_mercancia_GBP').rename_axis('semana').reset_index(),
              '28_serie_semanal_completa.csv',tables_dir)

    # Exportación de gráficas
    plot_ml_results(ml_metrics,figures_dir)
    plot_forecast_results(forecast_summary,figures_dir)
    plot_scenarios(scenario_table,figures_dir)
    graph_monthly(monthly,figures_dir);graph_validation(temporal,figures_dir)
    graph_final_validation(temporal,figures_dir);graph_segment(segments,figures_dir)
    graph_abc(abc,figures_dir);graph_cohort(cohort,figures_dir);graph_auc(auc,figures_dir)

    # Métricas y entorno de ejecución
    software={'python':platform.python_version(),
              'pandas':pd.__version__, 'numpy':np.__version__,
              'matplotlib':matplotlib.__version__, 'scikit_learn':version('scikit-learn')}
    reproducibilidad={'archivo_original':args.input.name,
         'sha256_original':audit['sha256'],
         'software':software,
         'semillas':{'modelos_clientes':2026,'bosque_pronostico':20261121,
                     'bootstrap_recompra':20261121},
         'cortes_clientes':CUTS,
         'entrenamiento':CUTS[:4], 'validacion':CUTS[4], 'prueba':CUTS[5],
         'metrica_AUC_PR_columna':'average_precision_score: precision promedio AP, no area trapezoidal',
         'alcance':'Evaluacion retrospectiva de un minorista; sin resultados causales'}
    with (metrics_dir/'29_entorno_reproducibilidad.json').open('w',encoding='utf-8') as f:
        json.dump(reproducibilidad,f,ensure_ascii=False,indent=2)
    with (metrics_dir/'indicadores_resultados.json').open('w',encoding='utf-8') as f:
        json.dump(json_ready(agg_canon),f,ensure_ascii=False,indent=2)
    # Resultados en consola
    print('\n' + '=' * 72)
    print('MODELOS DE RECOMPRA')
    print('=' * 72)
    print('Modelo seleccionado en validación:', selected_model)
    print(ml_metrics[['modelo','periodo','AUC_ROC','AUC_PR','Brier','precision_top20_pct','lift_top20']].round(4).to_string(index=False))
    print('DIFERENCIA AUC TEST vs RFM:',auc_difference)
    print('\n' + '=' * 72)
    print('PRONÓSTICOS DE VENTAS SEMANALES')
    print('=' * 72)
    print('Mejor método por WAPE medio:', best_forecast)
    print(forecast_summary.round(3).to_string(index=False))
    print('\n' + '=' * 72)
    print('ESCENARIOS ECONÓMICOS HIPOTÉTICOS')
    print('=' * 72)
    print('Clientes priorizados:', scenario_table.n_clientes_objetivo.iloc[0])
    print('Ticket histórico mediano (GBP):', scenario_table.ticket_mediano_historico_GBP.iloc[0])
    print('Escenarios evaluados:', len(scenario_table))

    print('\n' + '=' * 72)
    print('AUDITORÍA E INDICADORES DESCRIPTIVOS')
    print('=' * 72)
    print(json.dumps(json_ready({'auditoria':audit,'segunda_compra':cohort_metrics,'concentracion':concentration_metrics,
       'segmentos':segments[['Segmento','clientes','pct_clientes','pct_gasto']].to_dict(orient='records'),
       'comparacion_cortes':compare.to_dict(orient='records'),'auc':auc.to_dict(orient='records'),'intervalos_auc':interval_auc.to_dict(orient='records'),'ABC':abc.to_dict(orient='records'),
       'prioridad':priority.to_dict(orient='records'), 'parejas_mayor_lift':pairs.head(3).to_dict(orient='records')}),indent=2,ensure_ascii=False))
    print('\n' + '=' * 72)
    print('ARCHIVOS GENERADOS')
    print('=' * 72)
    print(f'Tablas CSV: {tables_dir.resolve()}')
    print(f'Gráficas PNG: {figures_dir.resolve()}')
    print(f'Métricas JSON: {metrics_dir.resolve()}')



if __name__ == '__main__':
    main()
