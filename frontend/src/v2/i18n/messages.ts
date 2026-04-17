// Source-of-truth UI string catalog. Add a key here with both
// languages before using it in a component. The `Messages` type is
// derived from the English bundle so missing zh keys become type
// errors at build time.

export const en = {
  // Brand / chrome
  'nav.search_placeholder': 'Search models…',
  'nav.open_palette': 'Open command palette',
  'nav.updated_fmt': 'Updated {relative}',
  'nav.updated_tooltip_fmt': 'Pricing snapshot built {absolute}',
  'footer.tagline': 'Model Price · compare LLMs across every major provider',
  'footer.unit': 'Prices per 1M tokens',
  'footer.github': 'GitHub',

  // Hero
  'hero.title_prefix': 'Compare ',
  'hero.title_suffix': ' LLMs from every major provider.',
  'hero.subtitle':
    'Real pricing, real capabilities. Keyboard-first. Shareable links. Built for devs who read configs more than marketing pages.',
  'hero.search_placeholder': 'Search by name, family, or model_id…',
  'hero.meta_matching_fmt': '{count} matching "{query}" of {total}',
  'hero.meta_shown_fmt': '{count} of {total} shown',

  // Filter bar
  'filter.all_makers': 'All makers',
  'filter.all_families': 'All families',
  'filter.sort.name': 'Name',
  'filter.sort.input': 'Input $',
  'filter.sort.output': 'Output $',
  'filter.sort.context': 'Context',
  'filter.cap.all': 'All',
  'cap.text': 'Text',
  'cap.vision': 'Vision',
  'cap.audio': 'Audio',
  'cap.tool_use': 'Tools',
  'cap.reasoning': 'Reasoning',
  'cap.function_calling': 'Functions',
  'cap.image_generation': 'Images',
  'cap.embedding': 'Embedding',

  // Table
  'table.col.model': 'Model',
  'table.col.context': 'Context',
  'table.col.input': 'Input / M',
  'table.col.output': 'Output / M',
  'table.col.capabilities': 'Capabilities',
  'table.empty': 'No models match your filters.',
  'table.add_to_compare_fmt': 'Add to compare ({count}/{max})',
  'table.remove_from_compare': 'Remove from compare',
  'table.compare_full_fmt': 'Compare is full — max {max} models',

  // Drawer / detail
  'detail.open_full_page': 'Open full page ↗',
  'detail.back_to_home': '← All models',
  'detail.not_found_fmt': 'Model "{slug}" not found.',
  'detail.loading': 'Loading…',
  'detail.copy_model_id': 'Copy model_id',
  'detail.copied': '✓ Copied!',
  'detail.add_to_compare': '+ Add to compare',
  'detail.in_compare': '✓ In compare',
  'detail.capabilities': 'Capabilities',
  'detail.modality_input': 'Input',
  'detail.modality_output': 'Output',
  'detail.pricing_across_providers': 'Pricing across providers',
  'detail.col.provider': 'Provider',
  'detail.col.input': 'Input',
  'detail.col.output': 'Output',
  'detail.col.cache_read': 'Cache read',
  'detail.col.batch_in': 'Batch in',
  'detail.primary_tag': 'primary',
  'detail.source_via_litellm': 'via LiteLLM',
  'detail.source_via_litellm_tooltip':
    "Pricing mirrored through the LiteLLM community registry, which tracks this provider's official prices.",
  'detail.source_placeholder': 'placeholder',
  'detail.source_placeholder_tooltip':
    'No direct provider offering yet; reference pricing from LiteLLM.',
  'detail.alternatives': 'Same tier, cheaper',
  'detail.context_suffix': 'context',
  'detail.max_output_suffix': 'max output',
  'detail.official_label': 'For full details:',
  'detail.official_pricing_fmt': '{maker} pricing ↗',
  'detail.official_docs_fmt': '{maker} docs ↗',
  'detail.litellm_source': 'source: LiteLLM ↗',

  // Alternatives card
  'alt.match_fmt': '{pct} match',
  'alt.input': 'Input',
  'alt.output': 'Output',

  // Compare page
  'compare.back_to_home': '← All models',
  'compare.heading_fmt': 'Compare {count} models',
  'compare.missing_fmt': 'Missing: {ids}',
  'compare.row.family': 'Family',
  'compare.row.context': 'Context',
  'compare.row.max_output': 'Max output',
  'compare.row.input': 'Input / M',
  'compare.row.output': 'Output / M',
  'compare.row.cache_read': 'Cache read',
  'compare.row.batch_in': 'Batch input',
  'compare.row.primary_provider': 'Primary provider',
  'compare.row.capabilities': 'Capabilities',
  'compare.remove': 'Remove',
  'compare.remove_tooltip': 'Remove from comparison',
  'compare.shared_fmt': 'Shared capabilities: {caps}',
  'compare.none_valid': 'No valid models in the comparison. Try selecting some again.',
  'compare.failed_fmt': 'Failed to load comparison. {error}',

  // Basket
  'basket.in_compare': 'in compare',
  'basket.count_fmt': '{count}/{max} in compare',
  'basket.hint_fmt': 'Up to {max} models',
  'basket.clear': 'Clear',
  'basket.go': 'Compare →',
  'basket.remove_fmt': 'Remove {slug}',

  // Command palette
  'palette.placeholder': 'Type a model name…',
  'palette.empty': 'No matches. Try a family name.',
  'palette.nav': 'navigate',
  'palette.open': 'open',
  'palette.copy_id': 'copy id',
  'palette.compare': 'compare',

  // Errors
  'error.backend_unreachable': 'Backend unreachable. Start uvicorn on :8000.',

  // Theme toggle
  'theme.dark': 'Dark',
  'theme.light': 'Light',
  'theme.system': 'System',
  'theme.next_fmt': 'Switch to {next}',

  // Locale toggle
  'locale.en': 'EN',
  'locale.zh': '中',
  'locale.switch_tooltip': 'Switch language',

  // Share
  'share.label': 'Share',
  'share.copy_link': 'Copy link',
  'share.link_copied': '✓ Link copied',
  'share.x_button': 'Share on X',
  'share.x_text_fmt': '{name} — {maker}. Compare LLM pricing at',

  // Export
  'export.csv': 'Export CSV',
  'export.csv_tooltip': 'Download current filtered list as CSV (opens in Excel)',
  'export.count_fmt': 'Export {count} models',
} as const;

export type MessageKey = keyof typeof en;

export const zh: Record<MessageKey, string> = {
  'nav.search_placeholder': '搜索模型…',
  'nav.open_palette': '打开命令面板',
  'nav.updated_fmt': '数据更新于 {relative}',
  'nav.updated_tooltip_fmt': '定价快照生成于 {absolute}',
  'footer.tagline': 'Model Price · 一站对比主流 LLM 定价',
  'footer.unit': '价格按每百万 token 计',
  'footer.github': 'GitHub',

  'hero.title_prefix': '对比 ',
  'hero.title_suffix': ' 个主流大模型,一站到底。',
  'hero.subtitle':
    '真实价格,真实能力。键盘优先,URL 可分享。为每天读 config 多过读官网的开发者打造。',
  'hero.search_placeholder': '按名称、系列或 model_id 搜索…',
  'hero.meta_matching_fmt': '{count} 个模型匹配 "{query}",共 {total}',
  'hero.meta_shown_fmt': '已显示 {count} / {total}',

  'filter.all_makers': '所有厂商',
  'filter.all_families': '所有系列',
  'filter.sort.name': '名称',
  'filter.sort.input': '输入价',
  'filter.sort.output': '输出价',
  'filter.sort.context': '上下文',
  'filter.cap.all': '全部',
  'cap.text': '文本',
  'cap.vision': '视觉',
  'cap.audio': '音频',
  'cap.tool_use': '工具',
  'cap.reasoning': '推理',
  'cap.function_calling': '函数调用',
  'cap.image_generation': '图像生成',
  'cap.embedding': '向量',

  'table.col.model': '模型',
  'table.col.context': '上下文',
  'table.col.input': '输入 / M',
  'table.col.output': '输出 / M',
  'table.col.capabilities': '能力',
  'table.empty': '没有模型匹配当前筛选条件。',
  'table.add_to_compare_fmt': '加入对比 ({count}/{max})',
  'table.remove_from_compare': '移出对比',
  'table.compare_full_fmt': '对比已满 — 最多 {max} 个模型',

  'detail.open_full_page': '打开独立页 ↗',
  'detail.back_to_home': '← 返回列表',
  'detail.not_found_fmt': '找不到模型 "{slug}"。',
  'detail.loading': '加载中…',
  'detail.copy_model_id': '复制 model_id',
  'detail.copied': '✓ 已复制!',
  'detail.add_to_compare': '+ 加入对比',
  'detail.in_compare': '✓ 已在对比中',
  'detail.capabilities': '能力',
  'detail.modality_input': '输入',
  'detail.modality_output': '输出',
  'detail.pricing_across_providers': '各提供商报价',
  'detail.col.provider': '提供商',
  'detail.col.input': '输入',
  'detail.col.output': '输出',
  'detail.col.cache_read': '缓存读取',
  'detail.col.batch_in': '批处理输入',
  'detail.primary_tag': '首选',
  'detail.source_via_litellm': '数据源 LiteLLM',
  'detail.source_via_litellm_tooltip':
    '该报价经由 LiteLLM 社区注册表同步自该提供商的官方定价,并非直接从提供商 API 抓取。',
  'detail.source_placeholder': '占位',
  'detail.source_placeholder_tooltip':
    '暂无任何直连提供商报价,此处仅展示 LiteLLM 注册表里的参考价格。',
  'detail.alternatives': '同档更便宜',
  'detail.context_suffix': '上下文',
  'detail.max_output_suffix': '最大输出',
  'detail.official_label': '完整细节:',
  'detail.official_pricing_fmt': '{maker} 官方价 ↗',
  'detail.official_docs_fmt': '{maker} 文档 ↗',
  'detail.litellm_source': '数据源: LiteLLM ↗',

  'alt.match_fmt': '{pct} 匹配',
  'alt.input': '输入',
  'alt.output': '输出',

  'compare.back_to_home': '← 返回列表',
  'compare.heading_fmt': '对比 {count} 个模型',
  'compare.missing_fmt': '缺失: {ids}',
  'compare.row.family': '系列',
  'compare.row.context': '上下文',
  'compare.row.max_output': '最大输出',
  'compare.row.input': '输入 / M',
  'compare.row.output': '输出 / M',
  'compare.row.cache_read': '缓存读取',
  'compare.row.batch_in': '批处理输入',
  'compare.row.primary_provider': '首选提供商',
  'compare.row.capabilities': '能力',
  'compare.remove': '移除',
  'compare.remove_tooltip': '从对比中移除',
  'compare.shared_fmt': '共同能力: {caps}',
  'compare.none_valid': '对比中没有有效的模型,请重新选择。',
  'compare.failed_fmt': '加载对比失败。{error}',

  'basket.in_compare': '在对比中',
  'basket.count_fmt': '对比中 · {count}/{max}',
  'basket.hint_fmt': '最多 {max} 个模型',
  'basket.clear': '清空',
  'basket.go': '开始对比 →',
  'basket.remove_fmt': '移除 {slug}',

  'palette.placeholder': '输入模型名…',
  'palette.empty': '没有匹配。试试按系列名搜索。',
  'palette.nav': '上下选择',
  'palette.open': '打开',
  'palette.copy_id': '复制 ID',
  'palette.compare': '加入对比',

  'error.backend_unreachable': '后端无法连接,请在 :8000 启动 uvicorn。',

  'theme.dark': '暗色',
  'theme.light': '亮色',
  'theme.system': '跟随系统',
  'theme.next_fmt': '切换到{next}',

  'locale.en': 'EN',
  'locale.zh': '中',
  'locale.switch_tooltip': '切换语言',

  'share.label': '分享',
  'share.copy_link': '复制链接',
  'share.link_copied': '✓ 链接已复制',
  'share.x_button': '分享到 X',
  'share.x_text_fmt': '{name} — {maker}。对比 LLM 价格',

  'export.csv': '导出 CSV',
  'export.csv_tooltip': '导出当前筛选结果为 CSV（Excel 可直接打开）',
  'export.count_fmt': '导出 {count} 个模型',
};

export const MESSAGES = { en, zh } as const;
