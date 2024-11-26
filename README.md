# Projektowanie systemów informatyki medycznej (projekt)

# Wstęp

Celem projektu było zaproponowanie architektury systemu informatycznego, który pozwala na klasyfikację obrazów RTG klatki piersiowej z wykorzystaniem sieci neuronowych.  
System ma wspomagać lekarzy w diagnostyce chorób poprzez automatyczną klasyfikację obrazów, a także dostarczać informacji o tym, jakie cechy obrazu zostały wykorzystane do podjęcia decyzji.

Głównym założeniem projektu nie jest stworzenie dokładnych modeli, ale eksploracja zagadnienia Explainable AI (XAI) i możliwości implementacji mechanizmów XAI w aplikacjach dla użytkownika końcowego.

## Motywacja

Zastosowanie głębokich sieci neuronowych w medycynie ma ogromny potencjał, ale wymaga specjalnych rozwiązań związanych z bezpieczeństwem i etyką.  
W szczególności problematyczny jest brak zrozumienia mechanizmów decyzyjnych głębokich sieci neuronowych, co generuje dalsze problemy regulacyjne i etyczne.

Efektywne *XAI* może pomóc w budowaniu zaufania i transparentności modeli dla użytkowników końcowych (pacjentów i lekarzy), a co za tym idzie, znacząco przyspieszyć procesy diagnostyczne.  
Ponadto wykorzystanie modeli głębokich w medycynie otwiera nowe możliwości w przygotowaniu spersonalizowanych planów terapii, co może znacząco poprawić jakość życia pacjentów.

## Zakres projektu

Ze względu na ograniczenia czasowe projektu, zakres implementacji został znacząco zawężony, a głównym celem było przeanalizowanie możliwości i potrzeb użytkowników końcowych w zakresie wyjaśnialności modeli.  
W szczególności w zakres projektu wchodziły następujące zagadnienia:

- Architektura systemu informatycznego  
- Proof-Of-Concept (PoC) dla klasyfikacji obrazów RTG klatki piersiowej  
- Wizualizacja wyników klasyfikacji w końcowej aplikacji  
- Analiza wyzwań związanych z wyjaśnialnością modeli  