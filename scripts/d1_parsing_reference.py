"""Generate parser oracles from parsing.py in the pinned d1 checkpoint.
Usage: python scripts/d1_parsing_reference.py CHECKPOINT OUTPUT_JSON
"""
import importlib.util
import json
import sys
from pathlib import Path
from d1_reference import REVISION


def main():
    root = Path(sys.argv[1])
    metadata = root / '.cache/huggingface/download/parsing.py.metadata'
    if metadata.read_text().splitlines()[0] != REVISION:
        raise ValueError('parsing.py must come from the pinned d1 revision')
    spec = importlib.util.spec_from_file_location('upstream_d1_parsing', root / 'parsing.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    numbers = [
        ('1 234,50 EUR', '', None), ('$1,250.50', '', None), ('1,250', '', None),
        ('1,250', 'price $12.50', None), ('1.250', 'language=de', None),
        ('1.234.567', '', None), ('12,50', '', None), ('12.5%', '', None),
        ('(£1,250.50)', '', None), ('- $ 125', '', None), ('($-125)', '', None),
        ('₹12,34,567.89', '', None), ('12,34,567.89', '', None),
        ('2.5 million', '', None), ('2 Billion', '', None), ('2 billion', '', None),
        ('2m', 'population count', None), ('2m', 'length', None),
        ('1\u202f234,50 €', '', None), ('1234', '', None), ('1,234', '', 'USD'),
        ('1,234', 'locale=fr', None), ('1,234', '12.50 and 12,50', None),
        ('1,234', '<?xml version="1.0"?>', None), ('١٢٣٤٫٥', '', None),
        ('١٢٣٤.٥', '', None), ('１２３４.５', '', None), ('−12.50', '', None),
        ('123456789012345678901234567890', '', None), ('0.0000001', '', None),
        ('one hundred', '', None), ('1e3', '', None), ('2万', '', None), ('3 crore', '', None),
    ]
    dates = [
        ('03/04/2026', ''), ('03/04/2026', 'day/month/year'), ('03/04/2026', 'locale=en-US'),
        ('03/04/2026', 'locale=en-GB'), ('03/04/2026', '13/04/2026'),
        ('03/04/2026', 'day/month/year and month/day/year'), ('13/04/2026', ''),
        ('2024-02-29', ''), ('2026-02-29', ''), ('03/04/26', ''),
        ('03/04/26', 'year 2026-11-04; dd/mm/yyyy'), ('03/04/26', '2026-11-04 and 1926-11-04'),
        ('2026年10月20日', ''), ('2026년 10월 20일', ''), ('Tue, 20 Oct 2026', ''),
        ('20 octobre 2026', ''), ('20 października 2026', ''), ('20 октября 2026', ''),
        ('segunda-feira, 20 de outubro de 2026', ''), ('October 20th, 2026', ''),
        ('2026-11-04T9:10', ''), ('2026-11-04 09:10:11.123Z', ''),
        ('2026-11-04T24:00', ''), ('2026-11-04T09:10:11+02:00', ''),
        ('２０２６-１１-０４', ''), ('yesterday', ''), ('20 Oct', ''),
    ]
    result = {'numbers': [dict(text=t, context=c, unit=u, expected=module.parse_number(t,c,unit=u),
                               written_unit=module.written_unit(t)) for t,c,u in numbers],
              'dates': [dict(text=t,context=c,expected=module.parse_date(t,c)[0]) for t,c in dates]}
    Path(sys.argv[2]).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    main()
