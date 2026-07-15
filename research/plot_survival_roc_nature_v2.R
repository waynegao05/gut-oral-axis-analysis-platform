suppressPackageStartupMessages({
  library(ggplot2)
  library(jsonlite)
  library(patchwork)
  library(scales)
})

STRICT_DARK <- "#484878"
STRICT_MID <- "#7884B4"
STRICT_SOFT <- "#DCE2F2"
HERO_DARK <- "#B35F7D"
HERO_SOFT <- "#F0C0CC"
NEUTRAL_DARK <- "#3F3F3F"
NEUTRAL_MID <- "#858585"
NEUTRAL_LIGHT <- "#D8D8D8"
CONTROL_BLUE <- "#9CB4CF"
CASE_ROSE <- "#D49AB0"
CHANCE_ROSE <- "#C7A8A8"

parse_args <- function(args) {
  values <- list(
    input_dir = "outputs/current_mainline_v2/survival_auc_v2",
    output_dir = "outputs/current_mainline_v2/survival_auc_v2/nature_roc_composite_r"
  )
  index <- 1
  while (index <= length(args)) {
    key <- args[[index]]
    if (key == "--input-dir" && index < length(args)) {
      values$input_dir <- args[[index + 1]]
      index <- index + 2
    } else if (key == "--output-dir" && index < length(args)) {
      values$output_dir <- args[[index + 1]]
      index <- index + 2
    } else {
      stop(sprintf("Unknown or incomplete argument: %s", key))
    }
  }
  values
}

read_report <- function(input_dir, filename) {
  path <- file.path(input_dir, filename)
  if (!file.exists(path)) {
    stop(sprintf("Required report does not exist: %s", path))
  }
  fromJSON(path, simplifyVector = FALSE)
}

find_horizon_row <- function(report, horizon) {
  matches <- Filter(
    function(row) isTRUE(all.equal(as.numeric(row$horizon), as.numeric(horizon))),
    report$horizons
  )
  if (length(matches) != 1) {
    stop(sprintf("Expected one row for horizon %s, found %d", horizon, length(matches)))
  }
  matches[[1]]
}

roc_frame <- function(row) {
  data.frame(
    fpr = as.numeric(unlist(row$false_positive_rate)),
    tpr = as.numeric(unlist(row$true_positive_rate))
  )
}

interpolate_roc <- function(row, grid) {
  curve <- roc_frame(row)
  collapsed <- aggregate(tpr ~ fpr, data = curve, FUN = max)
  approx(
    x = collapsed$fpr,
    y = collapsed$tpr,
    xout = grid,
    method = "linear",
    ties = max,
    rule = 2
  )$y
}

auc_vector <- function(report, horizons) {
  vapply(
    horizons,
    function(horizon) as.numeric(find_horizon_row(report, horizon)$auc),
    numeric(1)
  )
}

theme_nature <- function() {
  theme_classic(base_size = 6.8, base_family = "Arial") +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = NEUTRAL_DARK),
      axis.ticks = element_line(linewidth = 0.35, colour = NEUTRAL_DARK),
      axis.ticks.length = grid::unit(1.8, "pt"),
      axis.title = element_text(size = 6.8, colour = NEUTRAL_DARK),
      axis.text = element_text(size = 6.2, colour = NEUTRAL_DARK),
      plot.title = element_text(size = 7.2, face = "bold", margin = margin(b = 1.5)),
      plot.subtitle = element_text(size = 5.9, colour = NEUTRAL_MID, margin = margin(b = 3.5)),
      plot.tag = element_text(size = 8.2, face = "bold", colour = "black"),
      plot.tag.position = "topleft",
      plot.margin = margin(3, 4, 3, 3, unit = "pt"),
      legend.title = element_blank(),
      legend.text = element_text(size = 5.8),
      legend.key.height = grid::unit(5.5, "pt"),
      legend.key.width = grid::unit(12, "pt"),
      legend.spacing.x = grid::unit(2, "pt"),
      legend.margin = margin(0, 0, 0, 0),
      legend.box.margin = margin(0, 0, 0, 0),
      panel.grid = element_blank()
    )
}

make_roc_panel <- function(
    horizon,
    tag,
    row42,
    row43,
    champion_row,
    grid,
    source_rows) {
  tpr42 <- interpolate_roc(row42, grid)
  tpr43 <- interpolate_roc(row43, grid)
  curve42 <- roc_frame(row42)
  curve43 <- roc_frame(row43)
  strict <- data.frame(
    fpr = grid,
    mean = (tpr42 + tpr43) / 2,
    low = pmin(tpr42, tpr43),
    high = pmax(tpr42, tpr43)
  )
  champion <- roc_frame(champion_row)
  chance <- data.frame(fpr = c(0, 1), tpr = c(0, 1))
  strict_mean_auc <- mean(c(as.numeric(row42$auc), as.numeric(row43$auc)))
  champion_auc <- as.numeric(champion_row$auc)

  source_rows[[length(source_rows) + 1]] <- data.frame(
    panel = tag,
    horizon = horizon,
    series = "strict_split42",
    fpr = curve42$fpr,
    tpr = curve42$tpr,
    auc = as.numeric(row42$auc)
  )
  source_rows[[length(source_rows) + 1]] <- data.frame(
    panel = tag,
    horizon = horizon,
    series = "strict_split43",
    fpr = curve43$fpr,
    tpr = curve43$tpr,
    auc = as.numeric(row43$auc)
  )
  source_rows[[length(source_rows) + 1]] <- data.frame(
    panel = tag,
    horizon = horizon,
    series = "strict_cross_split_mean",
    fpr = grid,
    tpr = strict$mean,
    auc = strict_mean_auc
  )
  source_rows[[length(source_rows) + 1]] <- data.frame(
    panel = tag,
    horizon = horizon,
    series = "exploratory_single_split_best",
    fpr = champion$fpr,
    tpr = champion$tpr,
    auc = champion_auc
  )

  panel <- ggplot() +
    geom_step(
      data = curve42,
      aes(x = fpr, y = tpr, colour = "Strict split42", linetype = "Strict split42"),
      linewidth = 0.48,
      direction = "hv"
    ) +
    geom_step(
      data = curve43,
      aes(x = fpr, y = tpr, colour = "Strict split43", linetype = "Strict split43"),
      linewidth = 0.48,
      direction = "hv"
    ) +
    geom_step(
      data = champion,
      aes(
        x = fpr,
        y = tpr,
        colour = "Exploratory single-split best",
        linetype = "Exploratory single-split best"
      ),
      linewidth = 0.52,
      direction = "hv"
    ) +
    geom_line(
      data = chance,
      aes(x = fpr, y = tpr, colour = "Chance", linetype = "Chance"),
      linewidth = 0.38
    ) +
    scale_colour_manual(
      name = NULL,
      values = c(
        "Strict split42" = NEUTRAL_DARK,
        "Strict split43" = STRICT_MID,
        "Exploratory single-split best" = HERO_DARK,
        "Chance" = CHANCE_ROSE
      ),
      breaks = c("Strict split42", "Strict split43", "Exploratory single-split best", "Chance")
    ) +
    scale_linetype_manual(
      name = NULL,
      values = c(
        "Strict split42" = "solid",
        "Strict split43" = "solid",
        "Exploratory single-split best" = "22",
        "Chance" = "44"
      ),
      breaks = c("Strict split42", "Strict split43", "Exploratory single-split best", "Chance")
    ) +
    scale_x_continuous(
      breaks = seq(0, 1, 0.2),
      labels = label_number(accuracy = 0.1),
      limits = c(0, 1),
      expand = c(0, 0)
    ) +
    scale_y_continuous(
      breaks = seq(0, 1, 0.2),
      labels = label_number(accuracy = 0.1),
      limits = c(0, 1),
      expand = c(0, 0)
    ) +
    coord_equal() +
    labs(
      tag = tag,
      title = sprintf("%s time units", horizon),
      subtitle = sprintf("Mean AUC %.3f | exploratory %.3f", strict_mean_auc, champion_auc),
      x = "False-positive rate",
      y = if (tag == "a") "True-positive rate" else NULL
    ) +
    theme_nature() +
    theme(
      axis.line = element_blank(),
      axis.ticks.length = grid::unit(-1.7, "pt"),
      axis.text.x = element_text(margin = margin(t = 3.0)),
      axis.text.y = element_text(margin = margin(r = 3.0)),
      panel.border = element_rect(fill = NA, colour = NEUTRAL_DARK, linewidth = 0.35)
    ) +
    guides(
      colour = guide_legend(order = 1, nrow = 1, byrow = TRUE),
      linetype = guide_legend(order = 1, nrow = 1, byrow = TRUE)
    )
  list(plot = panel, source_rows = source_rows)
}

save_figure <- function(plot, output_base, width_in, height_in) {
  svglite::svglite(paste0(output_base, ".svg"), width = width_in, height = height_in, bg = "white")
  print(plot)
  grDevices::dev.off()

  grDevices::cairo_pdf(
    paste0(output_base, ".pdf"),
    width = width_in,
    height = height_in,
    family = "Arial",
    onefile = TRUE
  )
  print(plot)
  grDevices::dev.off()

  ragg::agg_tiff(
    paste0(output_base, ".tiff"),
    width = width_in,
    height = height_in,
    units = "in",
    res = 600,
    background = "white",
    compression = "lzw"
  )
  print(plot)
  grDevices::dev.off()

  ragg::agg_png(
    paste0(output_base, ".png"),
    width = width_in,
    height = height_in,
    units = "in",
    res = 300,
    background = "white"
  )
  print(plot)
  grDevices::dev.off()
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  input_dir <- args$input_dir
  output_dir <- args$output_dir
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  strict42 <- read_report(input_dir, "split42_three_seed_roc.json")
  strict43 <- read_report(input_dir, "split43_three_seed_roc.json")
  champion <- read_report(input_dir, "split42_champion_roc.json")
  champion_reference <- read_report(input_dir, "split42_champion_reference_auc.json")
  strict43_reference <- read_report(input_dir, "split43_reference_auc.json")

  horizons <- sort(vapply(strict42$horizons, function(row) as.numeric(row$horizon), numeric(1)))
  if (!identical(
    horizons,
    sort(vapply(strict43$horizons, function(row) as.numeric(row$horizon), numeric(1)))
  )) {
    stop("Strict split reports do not use identical horizons.")
  }
  if (!identical(
    horizons,
    sort(vapply(champion$horizons, function(row) as.numeric(row$horizon), numeric(1)))
  )) {
    stop("Champion and strict reports do not use identical horizons.")
  }

  grid <- seq(0, 1, length.out = 401)
  source_rows <- list()
  roc_plots <- list()
  for (index in seq_along(horizons)) {
    horizon <- horizons[[index]]
    built <- make_roc_panel(
      horizon = horizon,
      tag = letters[[index]],
      row42 = find_horizon_row(strict42, horizon),
      row43 = find_horizon_row(strict43, horizon),
      champion_row = find_horizon_row(champion, horizon),
      grid = grid,
      source_rows = source_rows
    )
    roc_plots[[index]] <- built$plot
    source_rows <- built$source_rows
  }

  auc42 <- auc_vector(strict42, horizons)
  auc43 <- auc_vector(strict43, horizons)
  champion_auc <- auc_vector(champion, horizons)
  champion_reference_auc <- auc_vector(champion_reference, horizons)
  strict43_reference_auc <- auc_vector(strict43_reference, horizons)
  strict_mean_auc <- (auc42 + auc43) / 2

  auc_summary <- data.frame(
    horizon = horizons,
    horizon_factor = factor(as.character(horizons), levels = rev(as.character(horizons))),
    low = pmin(auc42, auc43),
    high = pmax(auc42, auc43),
    strict_mean = strict_mean_auc,
    exploratory = champion_auc
  )
  auc_points <- rbind(
    data.frame(
      horizon_factor = auc_summary$horizon_factor,
      value = auc_summary$strict_mean,
      series = "Strict mean"
    ),
    data.frame(
      horizon_factor = auc_summary$horizon_factor,
      value = auc_summary$exploratory,
      series = "Exploratory best"
    )
  )
  panel_d <- ggplot(auc_summary, aes(y = horizon_factor)) +
    geom_segment(
      aes(x = low, xend = high, yend = horizon_factor),
      colour = STRICT_SOFT,
      linewidth = 2.2,
      lineend = "round"
    ) +
    geom_point(
      data = auc_points,
      aes(x = value, fill = series, shape = series),
      size = 2.1,
      colour = "white",
      stroke = 0.4
    ) +
    scale_fill_manual(
      values = c("Strict mean" = STRICT_DARK, "Exploratory best" = HERO_DARK)
    ) +
    scale_shape_manual(values = c("Strict mean" = 21, "Exploratory best" = 23)) +
    scale_x_continuous(
      breaks = c(0.80, 0.82, 0.84, 0.86),
      labels = label_number(accuracy = 0.01),
      limits = c(0.79, 0.865)
    ) +
    labs(
      tag = "d",
      title = "Discrimination across horizons",
      subtitle = "Horizontal bars show the split 42-43 range",
      x = "Time-dependent AUC",
      y = "Time horizon"
    ) +
    theme_nature() +
    theme(
      legend.position = "top",
      legend.justification = "left",
      legend.box.just = "left"
    )

  delta_values <- rbind(
    data.frame(
      horizon = horizons,
      model = "Exploratory best",
      delta = (champion_auc - champion_reference_auc) * 1000
    ),
    data.frame(
      horizon = horizons,
      model = "Strict split43",
      delta = (auc43 - strict43_reference_auc) * 1000
    )
  )
  horizon_levels <- rev(as.character(horizons))
  delta_values$base_y <- as.numeric(factor(as.character(delta_values$horizon), levels = horizon_levels))
  delta_values$y <- delta_values$base_y + ifelse(delta_values$model == "Exploratory best", 0.13, -0.13)
  delta_values$label <- sprintf("%+.2f", delta_values$delta)
  panel_e <- ggplot(delta_values, aes(colour = model)) +
    geom_vline(xintercept = 0, colour = NEUTRAL_MID, linewidth = 0.4, linetype = "dashed") +
    geom_segment(aes(x = 0, xend = delta, y = y, yend = y), linewidth = 0.7) +
    geom_point(aes(x = delta, y = y), size = 1.7) +
    geom_text(
      aes(x = delta, y = y, label = label, hjust = ifelse(delta >= 0, -0.18, 1.18)),
      size = 1.9,
      show.legend = FALSE
    ) +
    scale_colour_manual(
      values = c("Exploratory best" = HERO_DARK, "Strict split43" = STRICT_DARK)
    ) +
    scale_x_continuous(
      breaks = c(0, 0.5, 1.0, 1.5),
      limits = c(-0.28, 1.72)
    ) +
    scale_y_continuous(breaks = seq_along(horizon_levels), labels = horizon_levels) +
    labs(
      tag = "e",
      title = "Increment over reference",
      subtitle = "Positive values favour the refined risk score",
      x = expression(Delta * "AUC" ~ (x10^-3)),
      y = "Time horizon"
    ) +
    coord_cartesian(clip = "off") +
    theme_nature() +
    theme(
      legend.position = "top",
      legend.justification = "left",
      legend.box.just = "left"
    )

  count_rows <- list()
  for (horizon in horizons) {
    row42 <- find_horizon_row(strict42, horizon)
    row43 <- find_horizon_row(strict43, horizon)
    count_rows[[length(count_rows) + 1]] <- data.frame(
      horizon = horizon,
      category = c("Cases", "Dynamic controls", "Censored"),
      value = c(
        mean(c(as.numeric(row42$num_cases), as.numeric(row43$num_cases))),
        mean(c(as.numeric(row42$num_controls), as.numeric(row43$num_controls))),
        mean(c(
          as.numeric(row42$num_excluded_censored),
          as.numeric(row43$num_excluded_censored)
        ))
      )
    )
  }
  counts <- do.call(rbind, count_rows)
  counts$horizon_factor <- factor(as.character(counts$horizon), levels = rev(as.character(horizons)))
  counts$category <- factor(
    counts$category,
    levels = c("Censored", "Dynamic controls", "Cases")
  )
  counts$label <- ifelse(counts$value >= 35, sprintf("%.0f", counts$value), "")
  panel_f <- ggplot(counts, aes(x = value, y = horizon_factor, fill = category)) +
    geom_col(width = 0.58, colour = "white", linewidth = 0.25) +
    geom_text(
      aes(label = label),
      position = position_stack(vjust = 0.5),
      size = 1.9,
      colour = NEUTRAL_DARK
    ) +
    scale_fill_manual(
      values = c(
        "Cases" = CASE_ROSE,
        "Dynamic controls" = CONTROL_BLUE,
        "Censored" = NEUTRAL_LIGHT
      ),
      breaks = c("Cases", "Dynamic controls", "Censored")
    ) +
    scale_x_continuous(breaks = c(0, 360, 720), limits = c(0, 720), expand = c(0, 0)) +
    labs(
      tag = "f",
      title = "Risk-set composition",
      subtitle = "Mean sample count across split 42-43",
      x = "Samples",
      y = "Time horizon"
    ) +
    theme_nature() +
    theme(
      legend.position = "top",
      legend.justification = "left",
      legend.box.just = "left"
    )

  top_row <- wrap_plots(roc_plots, nrow = 1, guides = "collect") &
    theme(legend.position = "top")
  bottom_row <- panel_d + panel_e + panel_f + plot_layout(widths = c(1.05, 1, 1.05))
  composite <- top_row / bottom_row + plot_layout(heights = c(1.12, 1.0))

  output_base <- file.path(output_dir, "figure_survival_roc_nature_r")
  width_in <- 183 / 25.4
  height_in <- 150 / 25.4
  save_figure(composite, output_base, width_in, height_in)

  write.csv(
    do.call(rbind, source_rows),
    file.path(output_dir, "source_data_roc.csv"),
    row.names = FALSE
  )
  summary_source <- rbind(
    data.frame(
      panel = "d",
      horizon = horizons,
      metric = "auc_strict_split42",
      series = "strict_split42",
      value = auc42
    ),
    data.frame(
      panel = "d",
      horizon = horizons,
      metric = "auc_strict_split43",
      series = "strict_split43",
      value = auc43
    ),
    data.frame(
      panel = "d",
      horizon = horizons,
      metric = "auc_strict_cross_split_mean",
      series = "strict_cross_split_mean",
      value = strict_mean_auc
    ),
    data.frame(
      panel = "d",
      horizon = horizons,
      metric = "auc_exploratory_single_split_best",
      series = "exploratory_single_split_best",
      value = champion_auc
    ),
    data.frame(
      panel = "e",
      horizon = delta_values$horizon,
      metric = "auc_delta_x1000",
      series = delta_values$model,
      value = delta_values$delta
    ),
    data.frame(
      panel = "f",
      horizon = counts$horizon,
      metric = "sample_count",
      series = as.character(counts$category),
      value = counts$value
    )
  )
  write.csv(summary_source, file.path(output_dir, "source_data_summary.csv"), row.names = FALSE)

  legend_text <- c(
    "**Figure X | Time-dependent discrimination and robustness of the survival-risk model.**",
    "",
    "**a-c,** Cumulative/dynamic receiver-operating-characteristic curves at 36, 60 and 84 time units. Dark grey and indigo curves show the strict split42 and split43 results, respectively; the rose dashed curve is the exploratory single-split best model. Cases experienced an observed event by the horizon, whereas dynamic controls remained event-free beyond it. Samples censored before the horizon were excluded, and cases were weighted using inverse probabilities of censoring estimated from the training data.",
    "",
    "**d,** Horizon-specific cumulative/dynamic AUC. Horizontal bars span split42 to split43; circles show their arithmetic mean and diamonds show the exploratory single-split best result.",
    "",
    "**e,** Change in AUC relative to each model's prespecified reference risk score. Values are displayed in units of 1e-3.",
    "",
    "**f,** Mean composition of cases, dynamic controls and samples censored before each horizon across split42 and split43 (n = 720 per split).",
    "",
    "The topology_v6 data are synthetic/noisy expanded research data and are not a final clinical benchmark. Cross-split summaries currently include two splits. ROC averaging used linear interpolation on a shared false-positive-rate grid; all plotted source data are supplied with the figure."
  )
  writeLines(legend_text, file.path(output_dir, "figure_legend.md"), useBytes = TRUE)

  cat(toJSON(
    list(
      backend = "R",
      width_mm = 183,
      height_mm = 150,
      files = c(
        paste0(output_base, ".png"),
        paste0(output_base, ".svg"),
        paste0(output_base, ".pdf"),
        paste0(output_base, ".tiff"),
        file.path(output_dir, "source_data_roc.csv"),
        file.path(output_dir, "source_data_summary.csv"),
        file.path(output_dir, "figure_legend.md")
      )
    ),
    pretty = TRUE,
    auto_unbox = TRUE
  ))
  cat("\n")
}

if (sys.nframe() == 0) {
  main()
}
