# Bağlamsal Modeller Kullanarak Metinsel Veri Çoğaltma Kütüphanesi

## Odaklanılan sorun: 

Kısıtlı metinsel veri kümelerinin model geliştirmeleri için yetersiz kalması


## Sağlanan çözüm: 

Bağlamsal modellerden faydalanarak veri artırımı yapabilen doğal dil işleme kütüphanesi

## Uygulama Biçimi:

* Veri kümesinde yer alan cümlelere ait sözcüklerin rastgele maskelenerek yerine bağlamsal model (BERT) aracılığı ile olası sözcüklerin yerleştirilmesi

* İlgili değişiklikler sonucunda etiket bilgisinin korunması için anlamsal koşullar eklenmesi

* Artırılmış veri kümesi ile başarımın ölçümlenmesi ve sonuçların karşılaştırılması


## Kullanım

Uçtan uca bir örnek ["ornek_kullanim"](https://github.com/teksir-team/teksir/blob/master/ornek_kullanim.ipynb) notebook'unda gösterilmiştir.

```python
from augmentator import BertAugmentator

augmentation_config = {
    "model_name": "dbmdz/bert-base-turkish-cased",
    "frac": 0.2
}

bert_augmentator = BertAugmentator(augmentation_config=augmentation_config)

sentence = "TRABZON - Yurt dışı ve İstanbul 'da bazı transfer görüşmeleri yaptıktan sonra Trabzon'a gelen Süleyman Hurma, havalimanında basın mensuplarının sorularını yanıtladı."
augmented_sentence = bert_augmentator.augment(sentence)

"""
Original:
TRABZON - Yurt dışı ve İstanbul 'da bazı transfer görüşmeleri yaptıktan sonra Trabzon'a gelen Süleyman Hurma, havalimanında basın mensuplarının sorularını yanıtladı.
Augmented:
TRABZON - Yurt dışı ve Avrupa ' da çeşitli transfer görüşmeleri yaptıktan sonra Trabzon ' a gelen Özer Hurma , burada basın mensuplarının sorularını yanıtladı .
"""
```

## Örnekler

Aşağıda veri kümesinden rastgele seçilmiş örnekler üzerinde yapılan veri arttırım çalışmasına ait örnekler gösterilmiştir.

**Örnek 1:**

```
* "2004 yılında [MASK] Kanyon AVM metrekaresini 3 bin liradan satışa çıkarmıştık ."

{'score': 0.16715115308761597,
 'sequence': '[CLS] 2004 yılında İstanbul Kanyon AVM metrekaresini 3 bin liradan satışa çıkarmıştık. [SEP]',
 'token': 2673,
 'token_str': 'İstanbul'}

* "2004 yılında [İstanbul] Kanyon AVM metrekaresini 3 bin liradan satışa çıkarmıştık ."
```

**Örnek 2:**

```
* "Avrupa Birliği Bakanlığı'nda gerçekleşen kabul, basın mensuplarının [MASK] almasının ardından kapalı olarak devam etti."

{'score': 0.42132705450057983,
 'sequence': "[CLS] Avrupa Birliği Bakanlığı'nda gerçekleşen kabul, basın mensuplarının yerini almasının ardından kapalı olarak devam etti. [SEP]",
 'token': 5982,
 'token_str': 'yerini'}

 * Avrupa Birliği Bakanlığı'nda gerçekleşen kabul, basın mensuplarının yerini almasının ardından kapalı olarak devam etti.
```


## Veri Kümesi

Bu çalışmada [TTC-3600](https://github.com/denopas/TTC-3600) veri kümesi kullanılmıştır. Veri kümesi toplam 6 kategoriden (ekonomi, kültür-sanat, sağlık, siyaset, spor, teknoloji) 3600 doküman içermektedir.


## Sonuçlar

* Veri kümesi %80-%20 olacak şekilde eğitim ve test kümesi olarak ayrılmıştır. Ayrılan eğitim kümesinin %50'sine veri arttırımı uygulanmıştır.

* Veri kümesi üzerinde herhangi bir işlem yapılmadan önceki skorlar (Eğitim: 2880, Test: 720)

| label | precision | recall | f1-score |
|-------:|:-----------:|:--------:|:----------:|
| ekonomi|0,877|0,871|0,874 |
| kultursanat|0,894|0,924|0,909 |
| saglik|0,904|0,950|0,926 |
| siyaset|0,920|0,937|0,929 |
| spor|0,954|0,889|0,920 |
| teknoloji|0,904|0,879|0,891 |
|**macro**|**0,909**|**0,908**|**0,908**|

--- 

* Veri arttırımı yapıldıktan sonraki skorlar (Eğitim: 2880 + 1440 = 4320, Test: 720)

| label | precision | recall | f1-score |
|-------:|:-----------:|:--------:|:----------:|
| ekonomi|0,875|0,871|0,873 |
| kultursanat|0,893|0,916|0,905 |
| saglik|0,919|0,950|0,934 |
| siyaset|0,920|0,928|0,924 |
| spor|0,963|0,897|0,929 |
| teknoloji|0,896|0,882|0,889 |
|**macro**|**0,911**|**0,907**|**0,909**|


## Takım Üyeleri

* Sinan ÇALIŞIR
    * Sorumluluklar: Bağlamsal modellerin uygulanması ve veri kümelerinin hazırlanması


* Muhammed Emir KOÇAK
    * Sorumluluklar: Geliştirilecek modellerin tasarımı ve eğitimi


* Muhammed Furkan ÇANKAYA
    * Sorumluluklar: Geliştirilecek modellerin tasarımı ve eğitimi


* Yavuz TEZGEL (Herhangi bir çalışma yapmamıştır).

## Kaynaklar

* [BERT](https://arxiv.org/abs/1810.04805)
* [transformers](https://github.com/huggingface/transformers)
* [fastText](https://fasttext.cc/)