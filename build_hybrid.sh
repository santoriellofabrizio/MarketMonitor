#!/bin/bash
# ============================================================================
# Build script per MarketMonitor - Approccio Ibrido
# ============================================================================
#
# Questo script crea un deployment package completo:
# - Eseguibile PyInstaller (core MarketMonitor)
# - Strategie user_strategy come .py
# - Configurazioni
#
# Uso:
#   ./build_hybrid.sh
#   ./build_hybrid.sh --clean  (pulisce build precedenti)
#
# Output: deployment/ directory pronta per la distribuzione
# ============================================================================

set -e  # Exit on error

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}MarketMonitor - Hybrid Deployment Builder${NC}"
echo -e "${GREEN}============================================================================${NC}"

# Parse arguments
CLEAN=false
if [[ "$1" == "--clean" ]]; then
    CLEAN=true
    echo -e "${YELLOW}⚠️  Clean mode: rimuovo build precedenti${NC}"
fi

# 1. Clean (se richiesto)
if [ "$CLEAN" = true ]; then
    echo -e "${YELLOW}🧹 Pulizia build precedenti...${NC}"
    rm -rf build/ dist/ deployment/
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -delete
fi

# 2. Check dependencies
echo ""
echo -e "${GREEN}📦 Verifica dipendenze...${NC}"
if ! command -v pyinstaller &> /dev/null; then
    echo -e "${RED}❌ PyInstaller non trovato!${NC}"
    echo -e "${YELLOW}Installazione in corso...${NC}"
    pip install pyinstaller
fi

# 3. Build with PyInstaller
echo ""
echo -e "${GREEN}🔨 Build eseguibile con PyInstaller...${NC}"
pyinstaller run_strategy.spec

if [ ! -f "dist/run-strategy/run-strategy" ]; then
    echo -e "${RED}❌ Build fallito! Eseguibile non trovato.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Build completato!${NC}"

# 4. Create deployment package
echo ""
echo -e "${GREEN}📦 Creazione deployment package...${NC}"
rm -rf deployment/
mkdir -p deployment/

# Copy executable
echo "  - Copia eseguibile..."
cp -r dist/run-strategy deployment/

# Copy user_strategy (as .py files)
echo "  - Copia strategie (user_strategy)..."
cp -r user_strategy deployment/run-strategy/

# Copy config examples
echo "  - Copia configurazioni..."
mkdir -p deployment/run-strategy/etc/config
cp etc/config/config_hybrid_deployment_example.yaml deployment/run-strategy/etc/config/
cp etc/config/config_template.yaml deployment/run-strategy/etc/config/

# Create README for deployment
cat > deployment/run-strategy/README_DEPLOYMENT.txt << 'EOF'
============================================================================
MarketMonitor - Hybrid Deployment Package
============================================================================

CONTENUTO:
  - run-strategy          Eseguibile standalone
  - _internal/            Librerie Python bundled
  - user_strategy/        Strategie Python (modificabili)
  - etc/config/           Configurazioni YAML

USO:
  1. Modifica etc/config/config_hybrid_deployment_example.yaml
     - Cambia load_strategy_info per scegliere la strategia
     - Configura data sources (Bloomberg, Redis, etc.)

  2. Esegui:
     ./run-strategy etc/config/config_hybrid_deployment_example.yaml

     Oppure rinomina il config e usa l'alias:
     ./run-strategy my_config

STRATEGIE DISPONIBILI:
  - user_strategy/test_strategy/SimplePriceMonitorStrategy.py
  - user_strategy/test_strategy/TestTradeManagerStrategy.py
  - user_strategy/equity/LiveAnalysis/
  - user_strategy/fixed_income/

PERSONALIZZAZIONE:
  Le strategie in user_strategy/ sono file .py modificabili.
  Puoi editarli senza dover ricompilare l'eseguibile!

REQUISITI:
  - Nessuno! L'eseguibile include Python e tutte le dipendenze.
  - Bloomberg blpapi richiede installazione separata se usi Bloomberg.
  - Redis server se usi redis_data_distributor.

LOGS:
  I log vengono salvati in logs/

============================================================================
EOF

# 5. Create archive (optional)
echo ""
echo -e "${GREEN}📦 Creazione archivio...${NC}"
cd deployment
tar -czf MarketMonitor-hybrid-$(date +%Y%m%d).tar.gz run-strategy/
cd ..

# 6. Summary
echo ""
echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}✅ BUILD COMPLETATO!${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo ""
echo -e "📂 Output:"
echo -e "  - deployment/run-strategy/              (directory deployment)"
echo -e "  - deployment/MarketMonitor-hybrid-*.tar.gz  (archivio distribuibile)"
echo ""
echo -e "📊 Dimensioni:"
du -sh deployment/run-strategy 2>/dev/null || echo "  (calcolo dimensioni non disponibile)"
echo ""
echo -e "🚀 Prossimi passi:"
echo -e "  1. Copia deployment/run-strategy/ sul computer target"
echo -e "  2. Modifica etc/config/config_hybrid_deployment_example.yaml"
echo -e "  3. Esegui: ./run-strategy etc/config/..."
echo ""
echo -e "${GREEN}============================================================================${NC}"
